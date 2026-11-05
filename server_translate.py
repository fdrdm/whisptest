import gradio as gr
import stable_whisper
import requests
import re
import os
import logging
import time
import threading
import torch
import gc

import cola
import gpu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_translate")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "uso.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Whisper se carga por trabajo; gpu.trabajo() lo descarga al terminar (carga en ~10 s)
_whisper_model = None
_whisper_lock = threading.Lock()

def _unload_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("MODELO\twhisper turbo descargado")

def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            logging.info("MODELO\tcargando whisper turbo...")
            _whisper_model = stable_whisper.load_model('turbo')
            logging.info("MODELO\twhisper turbo cargado")
        return _whisper_model

gpu.registrar("whisper", _unload_whisper, lambda: _whisper_model is not None)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODELS = [
    "translategemma:latest",
]

PROMPT_TEMPLATE = """You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) subtitle translator for subtitle blocks. Keep it informal, natural, and suitable for YouTube across LATAM avoiding regionalisms.
{instrucciones_extra}.
The text is split into numbered subtitle blocks. Translate each one keeping the same numbering format. Use context from surrounding blocks to choose the best translation for ambiguous words.
Reply ONLY with the translated blocks in this exact format:
[1] translated text
[2] translated text
...

{text}"""

BATCH_SIZE = 10

IDIOMAS = {
    "English": "en",
    "Spanish": "es-MX",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
}

def segundos_a_srt(segundos):
    """Convierte los segundos al formato de tiempo que usan los archivos SRT"""
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos - int(segundos)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def procesar(input_file, *, ip, progress):
    if not input_file:
        return None

    # 1. Transcripción cruda (Gradio captura la barra de tqdm de Whisper)
    t0 = time.time()
    with gpu.trabajo("whisper"):
        model = _get_whisper()
        result = model.transcribe(
            input_file,
            progress_callback=lambda hecho, total: progress(hecho / total if total else 0, desc="Transcribiendo..."),
        )
    duracion_total = result.duration or 0
    idioma_detectado = result.language or "autodetectado"
    nombre_base = os.path.splitext(os.path.basename(input_file))[0]
    path_srt = os.path.join(OUTPUT_DIR, f"{nombre_base}_{idioma_detectado}.srt")

    # 2. Extraemos TODAS las palabras
    todas_las_palabras = []
    for segment in result.segments:
        for word in segment.words:
            todas_las_palabras.append(word)

    # 3. Armador de Loops
    MIN_PALABRAS = 4        # mínimo para cortar en fin de oración
    MIN_PALABRAS_SUAVE = 8  # mínimo para cortar antes de conjunción/preposición
    MAX_PALABRAS = 20       # tope duro
    MAX_DURACION = 15.0
    TOLERANCIA_ADELANTE = 5
    PAUSA_MINIMA_SUAVE = 0.25  # segundos de silencio para validar corte suave

    # Conjunciones y preposiciones que marcan inicio de cláusula
    CORTE_SUAVE = {
        # Conjunciones
        'and', 'or', 'but', 'so', 'yet', 'nor', 'because', 'although', 'though',
        'while', 'when', 'if', 'since', 'unless', 'until', 'whether',
        'however', 'moreover', 'nevertheless', 'otherwise', 'whereas', 'meanwhile',
        # Preposiciones que inician cláusula
        'with', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
        'between', 'against', 'without', 'beyond', 'despite', 'albeit',
    }

    # Conjunciones que no deben quedar colgadas al final de un bloque
    CONJUNCIONES = {'and', 'or', 'but', 'so', 'yet', 'nor'}

    loops = []
    loop_actual = []
    total_palabras = len(todas_las_palabras)
    i = 0

    while i < total_palabras:
        word = todas_las_palabras[i]
        palabra = word.word.strip().lower().rstrip('.,!?:;')

        # Antes de agregar: cortar si el bloque anterior terminó en fin de oración
        # y la palabra nueva empieza con mayúscula (= oración nueva)
        if loop_actual and len(loop_actual) >= 2:
            ultima = loop_actual[-1].word.strip()
            primera_letra = word.word.strip()[:1]
            if ultima.endswith(('.', '?', '!')) and primera_letra.isupper():
                loops.append(loop_actual)
                loop_actual = []

        # Antes de agregar: si viene conjunción/preposición, el bloque tiene suficiente,
        # Y hay una pausa real en el audio (no cortar en medio de habla fluida)
        if loop_actual and len(loop_actual) >= MIN_PALABRAS_SUAVE and palabra in CORTE_SUAVE:
            pausa = word.start - loop_actual[-1].end
            if pausa >= PAUSA_MINIMA_SUAVE:
                # Si la última palabra del bloque es una conjunción suelta ("but", "and"...),
                # moverla al bloque nuevo para que no quede colgada
                ultima_palabra = loop_actual[-1].word.strip().lower().rstrip('.,!?:;')
                if ultima_palabra in CONJUNCIONES and len(loop_actual) > 1:
                    colgada = loop_actual.pop()
                    loops.append(loop_actual)
                    loop_actual = [colgada]
                else:
                    loops.append(loop_actual)
                    loop_actual = []

        loop_actual.append(word)

        cantidad_palabras = len(loop_actual)
        duracion = loop_actual[-1].end - loop_actual[0].start
        termina_oracion = word.word.strip().endswith(('.', '?', '!'))
        es_ultima = (i == total_palabras - 1)

        if es_ultima:
            loops.append(loop_actual)
            break

        cortar = False

        # A. Fin de oración con al menos MIN_PALABRAS
        if termina_oracion and cantidad_palabras >= MIN_PALABRAS:
            cortar = True

        # B. Tope duro de palabras o tiempo
        if not cortar and (cantidad_palabras >= MAX_PALABRAS or duracion >= MAX_DURACION):
            # Buscar atrás la última coma o fin de oración para cortar limpio
            corte_atras = -1
            for j in range(len(loop_actual) - 1, MIN_PALABRAS - 1, -1):
                if loop_actual[j].word.strip().endswith(('.', '?', '!', ',')):
                    corte_atras = j
                    break

            if corte_atras >= 0:
                # Cortar en la coma/punto y pasar el resto al siguiente bloque
                loops.append(loop_actual[:corte_atras + 1])
                loop_actual = loop_actual[corte_atras + 1:]
                i += 1
                continue
            else:
                # Sin coma ni punto atrás, buscar adelante fin de oración
                palabras_extra = 0
                for j in range(1, TOLERANCIA_ADELANTE + 1):
                    if i + j >= total_palabras:
                        break
                    w_adelante = todas_las_palabras[i + j]
                    if w_adelante.word.strip().endswith(('.', '?', '!')):
                        for k in range(1, j + 1):
                            loop_actual.append(todas_las_palabras[i + k])
                        palabras_extra = j
                        break
                i += palabras_extra
                cortar = True

        if cortar:
            loops.append(loop_actual)
            loop_actual = []

        i += 1

    # Log
    duracion_habla = loops[-1][-1].end if loops else 0
    duracion_proceso = time.time() - t0
    logging.info(f"TRANSCRIPCION\t{nombre_base}\t{idioma_detectado}\taudio={duracion_total:.0f}s\thabla={duracion_habla:.0f}s\tproceso={duracion_proceso:.0f}s\tloops={len(loops)}\tvram_pico={gpu.pico_gb():.1f}GB\tip={ip}")

    # 4. Escribir el archivo SRT
    with open(path_srt, "w", encoding="utf-8") as f:
        for i, loop in enumerate(loops):
            inicio_srt = segundos_a_srt(loop[0].start)
            fin_srt = segundos_a_srt(loop[-1].end)

            texto_loop = "".join([w.word for w in loop]).strip()

            f.write(f"{i + 1}\n")
            f.write(f"{inicio_srt} --> {fin_srt}\n")
            f.write(f"{texto_loop}\n\n")

    return path_srt

def parsear_srt(contenido_srt):
    """Parsea un archivo SRT y devuelve una lista de bloques (numero, tiempo, texto)"""
    bloques = []
    partes = re.split(r'\n\n+', contenido_srt.strip())
    for parte in partes:
        lineas = parte.strip().split('\n')
        if len(lineas) >= 3:
            numero = lineas[0]
            tiempo = lineas[1]
            texto = '\n'.join(lineas[2:])
            bloques.append((numero, tiempo, texto))
    return bloques

def traducir_lote(textos_numerados, source_lang, source_code, target_lang, target_code, instrucciones_extra="", modelo=None):
    """Envía un lote de textos numerados a TranslateGemma y devuelve las traducciones"""
    extra = ""
    if instrucciones_extra.strip():
        extra = instrucciones_extra.strip()

    bloque_texto = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(textos_numerados))

    prompt = PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        source_code=source_code,
        target_lang=target_lang,
        target_code=target_code,
        instrucciones_extra=extra,
        text=bloque_texto,
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": modelo or OLLAMA_MODELS[0],
        "prompt": prompt,
        "stream": False,
        "keep_alive": "15m",
        # Los lotes son chicos; sin esto hereda OLLAMA_CONTEXT_LENGTH=65536 y ocupa más VRAM
        "options": {"num_ctx": 4096},
    }, timeout=300)
    resp.raise_for_status()
    respuesta = resp.json()["response"].strip()

    # Parsear la respuesta: buscar [1] texto, [2] texto, etc.
    traducciones = {}
    for match in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[|\Z)', respuesta, re.DOTALL):
        num = int(match.group(1))
        texto = match.group(2).strip()
        traducciones[num] = texto

    # Devolver en orden, con fallback al texto original si falta alguno
    resultado = []
    for i, texto_original in enumerate(textos_numerados):
        resultado.append(traducciones.get(i + 1, f"[SIN TRADUCCION] {texto_original}"))
    return resultado

def traducir_srt(archivo_srt, idioma_origen, idioma_destino, instrucciones, modelo, *, ip, progress):
    if not archivo_srt:
        return None, "No se subió ningún archivo SRT."

    t0 = time.time()
    nombre_srt = os.path.basename(archivo_srt)

    with open(archivo_srt, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = parsear_srt(contenido)
    if not bloques:
        return None, "El archivo SRT está vacío o tiene formato inválido."

    source_lang = idioma_origen
    source_code = IDIOMAS.get(idioma_origen, "en")
    target_lang = idioma_destino
    target_code = IDIOMAS.get(idioma_destino, "es")

    bloques_traducidos = []
    total = len(bloques)
    total_lotes = (total + BATCH_SIZE - 1) // BATCH_SIZE

    gpu.preparar_ollama()
    try:
        for lote_idx in range(total_lotes):
            inicio = lote_idx * BATCH_SIZE
            fin = min(inicio + BATCH_SIZE, total)
            lote = bloques[inicio:fin]

            progress((lote_idx + 1) / total_lotes, desc=f"Traduciendo lote {lote_idx + 1}/{total_lotes} (bloques {inicio + 1}-{fin}/{total})")

            textos = [texto for _, _, texto in lote]
            try:
                traducciones = traducir_lote(textos, source_lang, source_code, target_lang, target_code, instrucciones, modelo)
            except Exception as e:
                traducciones = [f"[ERROR: {e}]"] * len(textos)

            for i, (numero, tiempo, _) in enumerate(lote):
                bloques_traducidos.append((numero, tiempo, traducciones[i]))
    finally:
        gpu.descargar_ollama(modelo or OLLAMA_MODELS[0])

    nombre_base = os.path.splitext(nombre_srt)[0]
    if nombre_base.endswith(f"_{source_code}"):
        nombre_salida = f"{nombre_base[:-len(source_code)-1]}_{target_code}.srt"
    else:
        nombre_salida = f"{nombre_base}_{target_code}.srt"
    path_salida = os.path.join(OUTPUT_DIR, nombre_salida)

    with open(path_salida, "w", encoding="utf-8") as f:
        for numero, tiempo, texto in bloques_traducidos:
            f.write(f"{numero}\n{tiempo}\n{texto}\n\n")

    duracion_proceso = time.time() - t0
    logging.info(f"TRADUCCION\t{nombre_srt}\t{source_lang}->{target_lang}\tbloques={total}\tproceso={duracion_proceso:.0f}s\tip={ip}")

    return path_salida, f"Traducción completada: {total} bloques en {total_lotes} lotes ({source_lang} -> {target_lang})"

def procesar_en_cola(input_file, request: gr.Request, progress=gr.Progress()):
    if not input_file:
        return None
    t = cola.encolar("Transcribir", os.path.basename(input_file), procesar, (input_file,), cola.ip(request))
    return cola.esperar(t, progress)

def traducir_en_cola(archivo_srt, idioma_origen, idioma_destino, instrucciones, modelo, request: gr.Request, progress=gr.Progress()):
    if not archivo_srt:
        return None, "No se subió ningún archivo SRT."
    t = cola.encolar("Traducir", os.path.basename(archivo_srt), traducir_srt,
                     (archivo_srt, idioma_origen, idioma_destino, instrucciones, modelo), cola.ip(request))
    return cola.esperar(t, progress)

# ===================== INTERFAZ =====================
# app.py arma la página unificada con estas piezas.

def build_tabs():
    with gr.Tab("Transcribir"):
        input_data = gr.File(
            label="Subí tu Video o Audio (.mp3, .wav, .m4a, .aac, .flac, .ogg, .mp4, .mkv, etc.)",
            file_types=[
                "video", "audio",
                ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
                ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"
            ],
            type="filepath",
        )

        btn_transcribir = gr.Button("Generar SRT de Loops", variant="primary")

        with gr.Row():
            output_srt = gr.File(label="Archivo SRT")

        btn_transcribir.click(
            fn=procesar_en_cola,
            inputs=[input_data],
            outputs=[output_srt],
            concurrency_limit=None,  # la GPU la serializa cola.py
        )

    with gr.Tab("Traducir SRT"):
        with gr.Row():
            input_srt = gr.File(label="Subí tu archivo SRT", file_types=[".srt"])

        with gr.Row():
            idioma_origen = gr.Dropdown(
                choices=list(IDIOMAS.keys()),
                value="English",
                label="Idioma origen",
            )
            idioma_destino = gr.Dropdown(
                choices=list(IDIOMAS.keys()),
                value="Spanish",
                label="Idioma destino",
            )

        modelo_ollama = gr.Dropdown(
            choices=OLLAMA_MODELS,
            value=OLLAMA_MODELS[0],
            label="Modelo",
        )

        instrucciones = gr.Textbox(
            label="Instrucciones extra (opcional)",
            placeholder='Ej: Translate "muffins" as "magdalenas". Use informal tone (tuteo).',
            lines=3,
        )

        btn_traducir = gr.Button("Traducir SRT", variant="primary")

        with gr.Row():
            output_traducido = gr.File(label="SRT Traducido")
            output_estado = gr.Textbox(label="Estado", interactive=False)

        btn_traducir.click(
            fn=traducir_en_cola,
            inputs=[input_srt, idioma_origen, idioma_destino, instrucciones, modelo_ollama],
            outputs=[output_traducido, output_estado],
            concurrency_limit=None,  # la GPU la serializa cola.py
        )
