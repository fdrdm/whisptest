import gradio as gr
import stable_whisper
import whisper
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
import hablantes
import loops
import nemotron

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

PROMPT_TEMPLATE = """You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) subtitle translator{proposito}.
{reglas}{instrucciones_extra}.
The text is split into numbered subtitle blocks. Translate each one keeping the same numbering format. Use context from surrounding blocks to choose the best translation for ambiguous words.
Reply ONLY with the translated blocks in this exact format:
[1] translated text
[2] translated text
...

{text}"""

# Español: doblaje neutro para toda Latinoamérica. Comparado en un banco de 40 frases contra el prompt
# anterior (que metía "vale", "tío", "coger", "estáis", "qué padre"): sin palabras de España ni de un solo país.
DESTINO_PROMPT = {"Spanish": "Latin American Spanish"}
PROPOSITO = {"Spanish": ' for a neutral Latin American dub ("español neutro" as used in professional dubbing for all of Latin America)'}
REGLAS = {"Spanish": """Rules:
- Use "ustedes" for plural "you", never "vosotros" or its verb forms (-áis, -éis, imperatives like "venid").
- Use "tú" for informal singular; never "usted" unless the speaker is clearly being formal.
- Spanish drops subject pronouns: don't add "ustedes", "tú" or "él" unless needed for clarity.
- Translate the meaning of expressions, not word by word (e.g. "No way!" as surprise → "¡No puede ser!").
- Never use Spain-only words: vale, guay, tío/tía (for dude), coger, móvil, ordenador, flipar, mola, venga, chaval, zumo, patatas, joder, hostia.
- Never use slang from a single country (Mexico: güey, chido, qué padre, órale, neta, ahorita; Argentina: che, boludo, pibe, dale; Chile, Colombia, etc.).
- Prefer neutral options: okay/all right → vary between "bien", "bueno", "muy bien", "está bien" and "de acuerdo"; cool/awesome → "genial" or "increíble"; dude/bro/man → "amigo" or omit it; guys → "chicos" or omit it; phone → "teléfono" or "celular"; car → "auto"; computer → "computadora"; juice → "jugo"; potatoes → "papas"; swear words → mild neutral equivalents ("diablos", "rayos", "maldición").
- Keep it informal and natural, like a YouTube video.
"""}

BATCH_SIZE = 10
TEMPERATURA = 0.2  # baja: traducciones más estables entre corridas

IDIOMAS = {
    "English": "en",
    "Spanish": "es-419",
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

def palabras_json(path_srt):
    """JSON con las palabras y tiempos de la transcripción que generó path_srt."""
    nombre = os.path.splitext(os.path.basename(path_srt))[0] + ".json"
    return os.path.join(OUTPUT_DIR, "palabras", nombre)

def _con_nemotron(audio, identificar, num_hablantes, progress):
    """Texto de Nemotron + tiempos alineando ese texto con Whisper. Devuelve (result, etiquetas o None)."""
    with gpu.trabajo("nemotron"):
        crudo = nemotron.transcribir(audio, nemotron.prompt(identificar, num_hablantes),
                                     lambda f, desc="": progress(0.8 * f, desc=desc))
        vram = nemotron.vram_gb()

    # Tramos (hablante, texto) según las etiquetas <spk:N>; el texto de antes de la primera va con ella
    partes = re.split(r"<spk:(\d+)>", crudo)
    tramos = [(partes[i], partes[i + 1].strip()) for i in range(1, len(partes) - 1, 2)]
    if partes[0].strip() or not tramos:
        tramos.insert(0, (tramos[0][0] if tramos else "0", partes[0].strip()))
    tramos = [(h, t) for h, t in tramos if t]
    if not tramos:
        raise gr.Error("Nemotron no encontró habla en el audio.")
    texto, limites = "", []  # limites: (carácter donde termina el tramo, hablante)
    for h, t in tramos:
        texto += (" " if texto else "") + t
        limites.append((len(texto), h))

    progress(0.8, desc="Sincronizando con el audio...")
    with gpu.trabajo("whisper"):
        model = _get_whisper()
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        result = model.align(
            audio, texto, language=max(probs, key=probs.get),
            progress_callback=lambda hecho, total: progress(0.8 + 0.2 * hecho / total if total else 0.8,
                                                            desc="Sincronizando con el audio..."),
        )
        del model

    etiquetas = None
    if identificar:
        # Hablante de cada palabra según en qué tramo cae su mitad; H1, H2... por orden de aparición
        orden, etiquetas, pos, k = {}, [], 0, 0
        for w in result.all_words():
            while k < len(limites) - 1 and pos + len(w.word) / 2 > limites[k][0]:
                k += 1
            etiquetas.append(orden.setdefault(limites[k][1], f"H{len(orden) + 1}"))
            pos += len(w.word)
    return result, etiquetas, vram


def procesar(input_file, ajustes=None, identificar=False, num_hablantes=None, motor="Whisper", *, ip, progress):
    if not input_file:
        return None

    # 1. Transcripción cruda (Gradio captura la barra de tqdm de Whisper)
    t0 = time.time()
    audio, etiquetas, extra = None, None, ""
    if motor == "Nemotron":
        audio = whisper.load_audio(input_file)
        result, etiquetas, vram_nemotron = _con_nemotron(audio, identificar, num_hablantes, progress)
        vram_pico = gpu.pico_gb()  # la alineación con Whisper
        extra = f"\tvram_nemotron={vram_nemotron:.1f}GB"
    else:
        parte_whisper = 0.8 if identificar else 1.0
        with gpu.trabajo("whisper"):
            model = _get_whisper()
            result = model.transcribe(
                input_file,
                progress_callback=lambda hecho, total: progress(parte_whisper * hecho / total if total else 0,
                                                                desc="Transcribiendo..."),
            )
            del model  # que gpu.trabajo pueda liberarlo antes de pyannote
        vram_pico = gpu.pico_gb()
    duracion_total = result.duration or 0
    idioma_detectado = result.language or "autodetectado"
    nombre_base = os.path.splitext(os.path.basename(input_file))[0]
    path_srt = os.path.join(OUTPUT_DIR, f"{nombre_base}_{idioma_detectado}.srt")

    # 2. Guardamos las palabras con sus tiempos para poder rearmar los loops sin volver a transcribir
    path_palabras = palabras_json(path_srt)
    os.makedirs(os.path.dirname(path_palabras), exist_ok=True)
    result.save_as_json(path_palabras)

    todas_las_palabras = [word for segment in result.segments for word in segment.words]

    # 3. Quién habla cada palabra (hablantes.py)
    path_hablantes = hablantes_json(path_srt)
    if os.path.exists(path_hablantes):
        os.remove(path_hablantes)  # de una transcripción anterior del mismo video
    cantidad = ""
    if identificar and todas_las_palabras:
        if etiquetas is None:  # Whisper: los hablantes salen de pyannote (Nemotron ya los trae)
            audio = whisper.load_audio(input_file)
            turnos = hablantes.diarizar(
                audio, num_hablantes,
                progress=lambda f, desc="": progress(parte_whisper + (1 - parte_whisper) * f, desc=desc),
            )
            etiquetas = hablantes.asignar(todas_las_palabras, turnos)
        cantidad = f"\thablantes={len(hablantes.guardar(path_hablantes, todas_las_palabras, etiquetas, audio))}"
        vram_pico = max(vram_pico, gpu.pico_gb())

    # 4. Loops y SRT
    lista_loops = _armar_y_escribir(path_srt, todas_las_palabras, ajustes)

    # Log
    duracion_habla = lista_loops[-1][-1].end if lista_loops else 0
    duracion_proceso = time.time() - t0
    logging.info(f"TRANSCRIPCION\t{nombre_base}\t{idioma_detectado}\tmotor={motor.lower()}\taudio={duracion_total:.0f}s\thabla={duracion_habla:.0f}s\tproceso={duracion_proceso:.0f}s\tloops={len(lista_loops)}{cantidad}\tvram_pico={vram_pico:.1f}GB{extra}\tip={ip}")

    return path_srt

def hablantes_json(path_srt):
    return palabras_json(path_srt)[:-len(".json")] + ".hablantes.json"

def _armar_y_escribir(path_srt, palabras, ajustes):
    """Arma los loops (sin mezclar hablantes si se identificaron) y escribe el SRT con los nombres."""
    datos = hablantes.cargar(hablantes_json(path_srt))
    if datos and len(datos["palabras"]) == len(palabras):
        con_hablante = loops.armar_por_hablante(palabras, datos["palabras"], ajustes)
        nombres = {h: d["nombre"] for h, d in datos["hablantes"].items()}
        lista_loops = [loop for _, loop in con_hablante]
        loops.escribir_srt(lista_loops, os.path.join(OUTPUT_DIR, os.path.basename(path_srt)),
                           loops.nombres_por_loop(con_hablante, nombres, datos.get("todos", True)))
    else:
        lista_loops = loops.armar(palabras, ajustes)
        loops.escribir_srt(lista_loops, os.path.join(OUTPUT_DIR, os.path.basename(path_srt)))
    return lista_loops

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
        target_lang=DESTINO_PROMPT.get(target_lang, target_lang),
        target_code=target_code,
        proposito=PROPOSITO.get(target_lang, " for subtitle blocks. Keep it informal and natural, like a YouTube video"),
        reglas=REGLAS.get(target_lang, ""),
        instrucciones_extra=extra,
        text=bloque_texto,
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": modelo or OLLAMA_MODELS[0],
        "prompt": prompt,
        "stream": False,
        "keep_alive": "15m",
        # Los lotes son chicos; sin esto hereda OLLAMA_CONTEXT_LENGTH=65536 y ocupa más VRAM
        "options": {"num_ctx": 4096, "temperature": TEMPERATURA},
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

def _separar_nombres(textos):
    """Saca el "Nombre: " de cada bloque para que el modelo no lo traduzca. Devuelve (prefijos, textos).

    Cuenta como nombre lo que está antes de ": " si tiene hasta 3 palabras y, en el archivo,
    algún nombre se repite (así "Note: ..." suelto en un SRT sin hablantes no se toca).
    """
    partes = [re.match(r"^([^:\n.?!]{1,40}): (.*)$", t, re.DOTALL) for t in textos]
    candidatos = [m.group(1) for m in partes if m and len(m.group(1).split()) <= 3]
    if not any(candidatos.count(c) >= 2 for c in candidatos):
        return [None] * len(textos), list(textos)
    validos = set(candidatos)
    prefijos = [m.group(1) if m and m.group(1) in validos else None for m in partes]
    return prefijos, [m.group(2) if p else t for m, p, t in zip(partes, prefijos, textos)]

# ===================== REPASO DE LA TRADUCCIÓN =====================
# El modelo (4B) traduce de a 10 bloques y a veces se saltea uno, corre la traducción un renglón
# o se le escapa un regionalismo. Medido en un video de 1328 bloques: 1 bloque perdido, 11 "Vale".

# Regionalismos que delatan un doblaje no neutro. "vale" solo como interjección ("Vale.", "vale,"):
# "vale la pena" o "¿cuánto vale?" son neutros. "tío/tía" solo si en inglés no dice uncle/aunt.
REGIONALISMOS = {"Spanish": re.compile(
    r"\b(guay|tíos?|tías?|coger|cogí|coge|cógel\w*|móvil|ordenador|molan?|chaval\w*|joder|hostias?|vosotr[oa]s|"
    r"vuestr[oa]s?|os|\w+áis|\w+éis|tronco|flipa\w*|güey|wey|órale|chid[oa]s?|neta|ahorita|boludo|pibe|"
    r"qué padre|vale(?=\s*(?:[.,!?…]|$)))\b", re.IGNORECASE)}
NO_REGIONALISMO = {"dieciséis", "veintiséis"}

def _regionalismos(texto_en, texto, idioma):
    patron = REGIONALISMOS.get(idioma)
    if not patron:
        return []
    halladas = []
    for m in patron.finditer(texto):
        w = m.group(0).lower()
        if w in NO_REGIONALISMO:
            continue
        if w.startswith("tí") and re.search(r"\b(uncle|aunt)", texto_en, re.IGNORECASE):
            continue
        if w == "vale" and re.search(r"\b(cuánto|cuanto|no|lo|que)\s+vale", texto, re.IGNORECASE):
            continue
        halladas.append(w)
    return halladas

def _corrimientos(originales, traducidas):
    """Índices de vecinos con largos raros en sentidos opuestos: señal de que se corrió un renglón."""
    proporcion = [len(b) / max(len(a), 1) for a, b in zip(originales, traducidas)]
    largos = [len(a) >= 12 for a in originales]
    sospechosos = set()
    for i in range(len(originales) - 1):
        if largos[i] and largos[i + 1] and (
            (proporcion[i] > 2 and proporcion[i + 1] < 0.5) or (proporcion[i] < 0.5 and proporcion[i + 1] > 2)):
            sospechosos |= {i, i + 1}
    return sospechosos

def _falta(texto):
    return texto.startswith(("[SIN TRADUCCION]", "[ERROR:"))

def _repasar(originales, traducidas, idiomas, instrucciones, modelo, progress):
    """Retraduce de a uno los bloques con problemas. Devuelve (traducidas, retraducidos, a_revisar)."""
    source_lang, source_code, target_lang, target_code = idiomas
    problemas = sorted({i for i, t in enumerate(traducidas) if _falta(t)}
                       | {i for i, (a, t) in enumerate(zip(originales, traducidas)) if _regionalismos(a, t, target_lang)}
                       | _corrimientos(originales, traducidas))
    traducidas = list(traducidas)
    for n, i in enumerate(problemas):
        progress(1.0, desc=f"Repasando bloques dudosos ({n + 1}/{len(problemas)})")
        anterior = originales[i - 1] if i > 0 else ""
        siguiente = originales[i + 1] if i + 1 < len(originales) else ""
        contexto = f'Context only, do not translate it: previous line "{anterior}", next line "{siguiente}"'
        extra = f"{instrucciones.strip()}\n{contexto}" if instrucciones.strip() else contexto
        try:
            nueva = traducir_lote([originales[i]], source_lang, source_code, target_lang, target_code, extra, modelo)[0]
        except Exception as e:
            nueva = f"[ERROR: {e}]"
        if not _falta(nueva):
            traducidas[i] = nueva
    if target_lang == "Spanish":
        # Si el modelo insiste con el "Vale" de España, la interjección neutra es "Bien"
        traducidas = [re.sub(r"\b(V|v)ale(?=\s*(?:[.,!?…]|$))",
                             lambda m: "Bien" if m.group(1) == "V" else "bien", t)
                      if "vale" in _regionalismos(a, t, target_lang) else t
                      for a, t in zip(originales, traducidas)]
    a_revisar = [i for i, (a, t) in enumerate(zip(originales, traducidas))
                 if _falta(t) or _regionalismos(a, t, target_lang)]
    return traducidas, problemas, a_revisar

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

    prefijos, textos_sin_nombre = _separar_nombres([texto for _, _, texto in bloques])
    traducidas = []
    total = len(bloques)
    total_lotes = (total + BATCH_SIZE - 1) // BATCH_SIZE

    gpu.preparar_ollama()
    try:
        for lote_idx in range(total_lotes):
            inicio = lote_idx * BATCH_SIZE
            fin = min(inicio + BATCH_SIZE, total)

            progress((lote_idx + 1) / total_lotes, desc=f"Traduciendo lote {lote_idx + 1}/{total_lotes} (bloques {inicio + 1}-{fin}/{total})")

            textos = textos_sin_nombre[inicio:fin]
            try:
                traducciones = traducir_lote(textos, source_lang, source_code, target_lang, target_code, instrucciones, modelo)
            except Exception as e:
                traducciones = [f"[ERROR: {e}]"] * len(textos)

            traducidas += traducciones

        traducidas, retraducidos, a_revisar = _repasar(
            textos_sin_nombre, traducidas, (source_lang, source_code, target_lang, target_code),
            instrucciones, modelo, progress)
    finally:
        gpu.descargar_ollama(modelo or OLLAMA_MODELS[0])

    nombre_base = os.path.splitext(nombre_srt)[0]
    sufijo = target_code.split("-")[0]  # es-419 → video_es.srt
    sufijo_origen = source_code.split("-")[0]
    if nombre_base.endswith(f"_{sufijo_origen}"):
        nombre_salida = f"{nombre_base[:-len(sufijo_origen)-1]}_{sufijo}.srt"
    else:
        nombre_salida = f"{nombre_base}_{sufijo}.srt"
    path_salida = os.path.join(OUTPUT_DIR, nombre_salida)

    with open(path_salida, "w", encoding="utf-8") as f:
        for (numero, tiempo, _), prefijo, texto in zip(bloques, prefijos, traducidas):
            f.write(f"{numero}\n{tiempo}\n{f'{prefijo}: {texto}' if prefijo else texto}\n\n")

    duracion_proceso = time.time() - t0
    logging.info(f"TRADUCCION\t{nombre_srt}\t{source_lang}->{target_lang}\tbloques={total}\t"
                 f"retraducidos={len(retraducidos)}\ta_revisar={len(a_revisar)}\tproceso={duracion_proceso:.0f}s\tip={ip}")

    estado = f"Traducción completada: {total} bloques en {total_lotes} lotes ({source_lang} -> {target_lang})"
    if retraducidos:
        estado += f". Se repasaron {len(retraducidos)} bloques dudosos"
    if a_revisar:
        estado += f". Revisá a mano los bloques {', '.join(bloques[i][0] for i in a_revisar)}"
    return path_salida, estado

def procesar_en_cola(input_file, motor, identificar, num_hablantes, max_palabras, max_segundos, min_palabras, pausa_suave,
                     request: gr.Request, progress=gr.Progress()):
    if not input_file:
        return None
    if motor == "Nemotron" and not nemotron.disponible():
        raise gr.Error("Nemotron no está instalado en este servidor (falta llama-server o el modelo): corré setup.sh.")
    ajustes = _ajustes(max_palabras, max_segundos, min_palabras, pausa_suave)
    cantidad = int(num_hablantes) if str(num_hablantes).isdigit() else None
    t = cola.encolar("Transcribir", os.path.basename(input_file), procesar,
                     (input_file, ajustes, identificar, cantidad, motor), cola.ip(request))
    return cola.esperar(t, progress)

# ===================== REARMAR LOOPS =====================
# No pasa por la cola: no usa GPU y tarda milisegundos.

AJUSTES_DEF = loops.Ajustes()

def _ajustes(max_palabras, max_segundos, min_palabras, pausa_suave):
    return loops.Ajustes(max_palabras=int(max_palabras), max_segundos=float(max_segundos),
                         min_palabras=int(min_palabras), pausa_suave=float(pausa_suave))

def transcripciones():
    """SRT que se pueden rearmar (tienen sus palabras guardadas), el más nuevo primero."""
    carpeta = os.path.join(OUTPUT_DIR, "palabras")
    if not os.path.isdir(carpeta):
        return []
    jsons = [f for f in os.listdir(carpeta) if f.endswith(".json")]
    jsons.sort(key=lambda f: os.path.getmtime(os.path.join(carpeta, f)), reverse=True)
    return [f[:-len(".json")] + ".srt" for f in jsons]

def _traducciones(nombre_srt):
    """Otros SRT del mismo video: traducciones hechas con los loops anteriores."""
    idioma = r"_[A-Za-z]{2,3}(?:-[A-Za-z]{2})?\.srt"
    m = re.fullmatch(r"(.*)" + idioma, nombre_srt)
    if not m:
        return []
    patron = re.escape(m.group(1)) + idioma
    return sorted(f for f in os.listdir(OUTPUT_DIR) if f != nombre_srt and re.fullmatch(patron, f))

def mostrar_srt(nombre_srt, aviso=""):
    """(archivo, resumen, texto) del SRT elegido."""
    path = os.path.join(OUTPUT_DIR, nombre_srt) if nombre_srt else None
    if not path or not os.path.exists(path):
        return None, "", ""
    with open(path, encoding="utf-8") as f:
        texto = f.read()
    return path, aviso + loops.resumen_srt(texto), texto

def rearmar(nombre_srt, max_palabras, max_segundos, min_palabras, pausa_suave, request: gr.Request):
    if not nombre_srt:
        raise gr.Error("Elegí una transcripción.")
    path_palabras = palabras_json(nombre_srt)
    if not os.path.exists(path_palabras):
        raise gr.Error("Esa transcripción no tiene las palabras guardadas: volvé a transcribir el video.")
    t0 = time.time()
    resultado = stable_whisper.WhisperResult(path_palabras)
    ajustes = _ajustes(max_palabras, max_segundos, min_palabras, pausa_suave)
    lista_loops = _armar_y_escribir(nombre_srt, [w for s in resultado.segments for w in s.words], ajustes)
    logging.info(
        f"REARMADO\t{nombre_srt}\tloops={len(lista_loops)}\tmax_palabras={ajustes.max_palabras}\t"
        f"max_segundos={ajustes.max_segundos:g}\tmin_palabras={ajustes.min_palabras}\t"
        f"pausa_suave={ajustes.pausa_suave:g}\tproceso={time.time() - t0:.1f}s\tip={cola.ip(request)}"
    )
    otros = _traducciones(nombre_srt)
    aviso = (f"⚠️ Ojo: {', '.join(otros)} se hizo con los loops anteriores, volvé a traducirlo.\n\n"
             if otros else "")
    return mostrar_srt(nombre_srt, aviso)

def traducir_en_cola(archivo_srt, idioma_origen, idioma_destino, instrucciones, modelo, request: gr.Request, progress=gr.Progress()):
    if not archivo_srt:
        return None, "No se subió ningún archivo SRT."
    t = cola.encolar("Traducir", os.path.basename(archivo_srt), traducir_srt,
                     (archivo_srt, idioma_origen, idioma_destino, instrucciones, modelo), cola.ip(request))
    return cola.esperar(t, progress)

# ===================== INTERFAZ =====================
# app.py arma la página unificada con estas piezas.

def build_tabs(demo):
    with gr.Tab("Transcribir") as tab_transcribir:
        input_data = gr.File(
            label="Subí tu Video o Audio (.mp3, .wav, .m4a, .aac, .flac, .ogg, .mp4, .mkv, etc.)",
            file_types=[
                "video", "audio",
                ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
                ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"
            ],
            type="filepath",
        )

        with gr.Row():
            motor = gr.Radio(["Whisper", "Nemotron"], value="Whisper", label="Motor",
                             info="Whisper: rápido. Nemotron: más lento (~2 min cada 10 de audio, hasta 1 hora), "
                                  "detecta los hablantes él mismo.")
            identificar = gr.Checkbox(True, label="Identificar hablantes",
                                      info="Pone \"Nombre: texto\" en el SRT (con Whisper usa pyannote, suma unos 15 s).")
            num_hablantes = gr.Dropdown(["Automático"] + [str(n) for n in range(1, 11)], value="Automático",
                                        label="Cantidad de hablantes", info="Si la sabés, mejora la precisión")

        d = AJUSTES_DEF
        with gr.Accordion("Ajustes de loops", open=False):
            with gr.Row():
                max_palabras = gr.Slider(5, 40, value=d.max_palabras, step=1, label="Máximo de palabras",
                                         info=f"Puede pasarse hasta {d.tolerancia_adelante} para cerrar la oración. Por defecto: {d.max_palabras}")
                max_segundos = gr.Slider(3, 30, value=d.max_segundos, step=0.5, label="Máximo de segundos",
                                         info=f"Puede pasarse un poco para cerrar la oración. Por defecto: {d.max_segundos:g}")
            with gr.Row():
                min_palabras = gr.Slider(1, 10, value=d.min_palabras, step=1, label="Mínimo para cortar",
                                         info=f"Palabras mínimas para cortar en un punto. Por defecto: {d.min_palabras}")
                pausa_suave = gr.Slider(0, 1, value=d.pausa_suave, step=0.05, label="Pausa para corte suave (s)",
                                        info=f'Silencio mínimo para cortar antes de "and", "but", "with"... '
                                             f'Por defecto: {str(d.pausa_suave).replace(".", ",")}')
            btn_por_defecto = gr.Button("Volver a los valores por defecto", size="sm", variant="secondary")
        ajustes = [max_palabras, max_segundos, min_palabras, pausa_suave]
        valores_def = [d.max_palabras, d.max_segundos, d.min_palabras, d.pausa_suave]
        # Cada persona ve los últimos ajustes que usó (se guardan en su navegador)
        ajustes_guardados = gr.BrowserState(valores_def, storage_key="whisptest_ajustes_loops")

        btn_transcribir = gr.Button("Generar SRT de Loops", variant="primary")

        gr.Markdown("### Resultado")
        with gr.Row():
            elegido = gr.Dropdown(choices=transcripciones(), value=None, label="Transcripción",
                                  info="La que acabás de hacer, o cualquier anterior", scale=3)
            btn_rearmar = gr.Button("Rearmar con estos ajustes", scale=1)
        resumen = gr.Markdown()
        refresco = gr.State(0)  # para redibujar el panel aunque se repita el mismo video

        @gr.render(inputs=[elegido, refresco])
        def panel_hablantes(nombre_srt, _):
            datos = hablantes.cargar(hablantes_json(nombre_srt)) if nombre_srt else None
            if not datos or not datos.get("hablantes"):
                return
            sugeridos = hablantes.nombres_usados(os.path.join(OUTPUT_DIR, "palabras"))
            gr.Markdown("### Hablantes\nEscuchá cada muestra, poné los nombres y aplicalos. "
                        "Si pyannote partió a una persona en dos, poneles el mismo nombre.")
            campos = []
            for h, info in datos["hablantes"].items():
                with gr.Row(equal_height=True):
                    gr.Audio(info["muestra"], type="filepath", interactive=False, show_label=False, scale=2)
                    frase = info["frase"] if len(info["frase"]) <= 140 else info["frase"][:140] + "…"
                    m, seg = divmod(int(info["segundos"]), 60)
                    with gr.Column(scale=3):
                        gr.Markdown(f"**Hablante {h[1:]}** · habla {f'{m}m {seg:02d}s' if m else f'{seg}s'}\n\n_{frase}_")
                    campos.append(gr.Dropdown(sugeridos, value=info["nombre"] or None, allow_custom_value=True,
                                              label="Nombre", interactive=True, scale=2))
            with gr.Row():
                todos = gr.Checkbox(datos.get("todos", True), label="Nombre en todos los loops", interactive=True,
                                    info="Destildado: solo cuando cambia el hablante o tras una pausa de más de 2 s")
                btn_aplicar = gr.Button("Aplicar nombres", variant="primary")

            def aplicar_nombres(todos_valor, *valores):
                nombres = dict(zip(datos["hablantes"], valores[:len(campos)]))
                hablantes.poner_nombres(hablantes_json(nombre_srt), nombres, todos_valor)
                return rearmar(nombre_srt, *valores[len(campos):], None)

            btn_aplicar.click(aplicar_nombres, [todos, *campos, *ajustes], [output_srt, resumen, vista])

        vista = gr.Textbox(label="Vista previa", lines=10, max_lines=10, interactive=False, buttons=["copy"])
        output_srt = gr.File(label="Archivo SRT", interactive=False)

        def al_terminar(path, n):
            nombre = os.path.basename(path) if path else None
            return gr.update(choices=transcripciones(), value=nombre), *mostrar_srt(nombre)[1:], n + 1

        btn_transcribir.click(
            fn=procesar_en_cola,
            inputs=[input_data, motor, identificar, num_hablantes, *ajustes],
            outputs=[output_srt],
            concurrency_limit=None,  # la GPU la serializa cola.py
        ).success(al_terminar, [output_srt, refresco], [elegido, resumen, vista, refresco])
        btn_rearmar.click(rearmar, [elegido, *ajustes], [output_srt, resumen, vista])
        elegido.input(mostrar_srt, elegido, [output_srt, resumen, vista])
        tab_transcribir.select(lambda: gr.update(choices=transcripciones()), None, elegido, show_progress="hidden")
        btn_por_defecto.click(lambda: valores_def, None, ajustes)
        demo.load(lambda guardados: list(guardados), ajustes_guardados, ajustes)
        gr.on([a.change for a in ajustes], lambda *valores: list(valores), ajustes, ajustes_guardados,
              show_progress="hidden")

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
