import gradio as gr
import stable_whisper
import whisper
import requests
import json
import re
import os
import logging
import time
import threading
import torch
import gc
from urllib.parse import quote

import cola
import gpu
import hablantes
import loops
import nemotron

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_translate")
DEEPINFRA = "Nemotron (DeepInfra)"
MOTORES = {"Whisper": "whisper", "Nemotron": "nemotron", DEEPINFRA: "deepinfra"}  # el sufijo de los archivos
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

def _idioma(model, audio):
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def _hache(hablantes_crudos):
    """Etiquetas H1, H2... por orden de aparición."""
    orden = {}
    return [orden.setdefault(h, f"H{len(orden) + 1}") for h in hablantes_crudos]


def _alinear(audio, tramos, progress, desde):
    """Tiempos para el texto de Nemotron alineándolo con Whisper.

    tramos: [(hablante, texto)] en orden. Devuelve (result, hablante crudo de cada palabra).
    """
    tramos = [(h, t.strip()) for h, t in tramos if t.strip()]
    if not tramos:
        raise gr.Error("Nemotron no encontró habla en el audio.")
    texto, limites = "", []  # limites: (carácter donde termina el tramo, hablante)
    for h, t in tramos:
        texto += (" " if texto else "") + t
        limites.append((len(texto), h))

    progress(desde, desc="Sincronizando con el audio...")
    with gpu.trabajo("whisper"):
        model = _get_whisper()
        # vad: la voz la detecta Silero y no el volumen; si no, el texto se estira sobre la música de
        # una intro sin letra (Futurama: las primeras líneas quedaban 40 s antes)
        result = model.align(
            audio, texto, language=_idioma(model, audio), vad=True,
            progress_callback=lambda hecho, total: progress(desde + (1 - desde) * hecho / total if total else desde,
                                                            desc="Sincronizando con el audio..."),
        )
        del model

    # Hablante de cada palabra según en qué tramo cae su mitad
    crudos, pos, k = [], 0, 0
    for w in result.all_words():
        while k < len(limites) - 1 and pos + len(w.word) / 2 > limites[k][0]:
            k += 1
        crudos.append(limites[k][1])
        pos += len(w.word)
    return result, crudos


def _por_hablante(crudo):
    """[(hablante, texto)] según las etiquetas <spk:N>; el texto de antes de la primera va con ella."""
    partes = re.split(r"<spk:(\d+)>", crudo)
    tramos = [(partes[i], partes[i + 1]) for i in range(1, len(partes) - 1, 2)]
    if partes[0].strip() or not tramos:
        tramos.insert(0, (tramos[0][0] if tramos else "0", partes[0]))
    return tramos


def _con_nemotron(audio, identificar, num_hablantes, progress):
    """Texto de Nemotron + tiempos alineando ese texto con Whisper. Devuelve (result, etiquetas o None, vram).

    Va en tramos de ~5 min (nemotron.TRAMO_SEG, los mismos que DeepInfra): los hablantes de cada tramo
    son distintos (al mismo personaje se le pone el mismo nombre en cada uno).
    """
    partes = nemotron.tramos(audio)
    tramos = []
    with gpu.trabajo("nemotron"):
        for k, (a, b) in enumerate(partes):
            def avance(f, desc="", k=k):
                progress(0.8 * (k + f) / len(partes), desc=desc + (f" (tramo {k + 1} de {len(partes)})" if len(partes) > 1 else ""))
            crudo = nemotron.transcribir(audio[a:b], nemotron.prompt(identificar, num_hablantes), avance)
            tramos += [(f"{k}:{h}", t) for h, t in _por_hablante(crudo)]
        vram = nemotron.vram_gb()
    result, crudos = _alinear(audio, tramos, progress, 0.8)
    return result, _hache(crudos) if identificar else None, vram


def _con_deepinfra(input_file, audio, identificar, num_hablantes, progress):
    """Nemotron en DeepInfra: ya trae tiempos y hablante por palabra; el idioma lo da el Whisper de DeepInfra.

    Nada corre local salvo que los tiempos vengan rotos. Devuelve (result, etiquetas o None, nota del log, respuesta cruda).
    """
    segundos = len(audio) / nemotron.SR
    if segundos > nemotron.MAX_SEGUNDOS:
        raise gr.Error(f"Nemotron acepta hasta {nemotron.MAX_SEGUNDOS // 60} minutos de audio; este dura "
                       f"{segundos / 60:.0f}. Usá Whisper para este archivo.")
    d = nemotron.deepinfra(audio, num_hablantes, lambda f, desc="": progress(0.9 * f, desc=desc))
    palabras = [w for w in d.get("words") or [] if w["word"].strip()]
    costo_total, nota = d.get("inference_status", {}).get("cost", 0), ""
    if not palabras:
        raise gr.Error("Nemotron no encontró habla en el audio.")

    if nemotron.tiempos_sanos(palabras):
        # Idioma con el Whisper de DeepInfra, sobre 30 s desde la mitad de lo hablado (al principio suele haber música)
        progress(0.9, desc="Detectando el idioma...")
        idioma, costo = nemotron.deepinfra_idioma(input_file, palabras[len(palabras) // 2]["start"])
        costo_total += costo
        # Un segmento por tramo seguido de un mismo hablante (los loops solo usan las palabras)
        tramos = []
        for w in palabras:
            p = {"word": " " + w["word"].strip(), "start": w["start"], "end": max(w["end"], w["start"]), "probability": 1.0}
            if tramos and tramos[-1][0] == w["speaker"]:
                tramos[-1][1].append(p)
            else:
                tramos.append((w["speaker"], [p]))
        segmentos = [{"start": ps[0]["start"], "end": ps[-1]["end"], "text": "".join(p["word"] for p in ps), "words": ps}
                     for _, ps in tramos]
        # Con dos voces a la vez las palabras se pisan: force_order acomoda los tiempos
        result = stable_whisper.WhisperResult({"language": idioma, "segments": segmentos}, force_order=True)
        crudos = [w["speaker"] for w in palabras]
    else:
        logging.info("DEEPINFRA\ttiempos rotos: se alinea el texto con Whisper")
        nota += "\talineado=whisper"
        tramos = []
        for w in palabras:
            if tramos and tramos[-1][0] == w["speaker"]:
                tramos[-1][1] += " " + w["word"]
            else:
                tramos.append([w["speaker"], w["word"]])
        result, crudos = _alinear(audio, tramos, progress, 0.9)
    return result, _hache(crudos) if identificar else None, f"\tcosto_usd={costo_total:.3f}{nota}", d


def procesar(input_file, ajustes=None, identificar=True, num_hablantes=None, motor="Whisper", *, ip, progress):
    if not input_file:
        return None

    # 1. Transcripción cruda (Gradio captura la barra de tqdm de Whisper)
    t0 = time.time()
    audio, etiquetas, extra, crudo = None, None, "", None
    if motor == "Nemotron":
        audio = whisper.load_audio(input_file)
        result, etiquetas, vram_nemotron = _con_nemotron(audio, identificar, num_hablantes, progress)
        vram_pico = gpu.pico_gb()  # la alineación con Whisper
        extra = f"\tvram_nemotron={vram_nemotron:.1f}GB"
    elif motor == DEEPINFRA:
        audio = whisper.load_audio(input_file)
        result, etiquetas, extra, crudo = _con_deepinfra(input_file, audio, identificar, num_hablantes, progress)
        vram_pico = gpu.pico_gb()
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
    duracion_total = len(audio) / nemotron.SR if audio is not None else (result.duration or 0)
    idioma_detectado = result.language or "autodetectado"
    nombre_base = os.path.splitext(os.path.basename(input_file))[0]
    # El motor va en el nombre (video_en.nemotron.srt) para poder comparar las versiones
    path_srt = os.path.join(OUTPUT_DIR, f"{nombre_base}_{idioma_detectado}.{MOTORES[motor]}.srt")

    # 2. Guardamos las palabras con sus tiempos para poder rearmar los loops sin volver a transcribir
    path_palabras = palabras_json(path_srt)
    os.makedirs(os.path.dirname(path_palabras), exist_ok=True)
    result.save_as_json(path_palabras)
    if crudo:  # la respuesta de DeepInfra tal cual, para revisar cuando sale mal
        with open(path_palabras[:-len(".json")] + ".crudo.json", "w", encoding="utf-8") as f:
            json.dump(crudo, f, ensure_ascii=False)

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
    logging.info(f"TRANSCRIPCION\t{nombre_base}\t{idioma_detectado}\tmotor={MOTORES[motor]}\taudio={duracion_total:.0f}s\thabla={duracion_habla:.0f}s\tproceso={duracion_proceso:.0f}s\tloops={len(lista_loops)}{cantidad}\tvram_pico={vram_pico:.1f}GB{extra}\tip={ip}")

    return salidas(path_srt)

def hablantes_json(path_srt):
    return palabras_json(path_srt)[:-len(".json")] + ".hablantes.json"

def srt_nombres(path_srt):
    """video_en.srt -> video_en.nombres.srt: los mismos loops con "Nombre: texto"."""
    return os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(path_srt))[0] + ".nombres.srt")

def salidas(path_srt):
    """El SRT limpio y, si hay hablantes, el de nombres."""
    path_srt = os.path.join(OUTPUT_DIR, os.path.basename(path_srt))
    return [p for p in (path_srt, srt_nombres(path_srt)) if os.path.exists(p)]

def _armar_y_escribir(path_srt, palabras, ajustes):
    """Arma los loops (sin mezclar hablantes si se identificaron) y escribe el SRT limpio y el de nombres."""
    datos = hablantes.cargar(hablantes_json(path_srt))
    if datos and len(datos["palabras"]) == len(palabras):
        con_hablante = loops.armar_por_hablante(palabras, datos["palabras"], ajustes)
        nombres = {h: d["nombre"] for h, d in datos["hablantes"].items()}
        lista_loops = [loop for _, loop in con_hablante]
        loops.escribir_srt(lista_loops, srt_nombres(path_srt),
                           loops.nombres_por_loop(con_hablante, nombres, datos.get("todos", True)))
    else:
        lista_loops = loops.armar(palabras, ajustes)
        if os.path.exists(srt_nombres(path_srt)):  # de una transcripción anterior con hablantes
            os.remove(srt_nombres(path_srt))
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

    # video_en.nemotron.nombres → video_es.nemotron.nombres (el motor y los nombres se conservan)
    nombre_base, extras = re.fullmatch(r"(.*?)((?:\.(?:whisper|nemotron|deepinfra|nombres))*)",
                                       os.path.splitext(nombre_srt)[0]).groups()
    sufijo = target_code.split("-")[0]  # es-419 → video_es.srt
    sufijo_origen = source_code.split("-")[0]
    if nombre_base.endswith(f"_{sufijo_origen}"):
        nombre_salida = f"{nombre_base[:-len(sufijo_origen)-1]}_{sufijo}{extras}.srt"
    else:
        nombre_salida = f"{nombre_base}_{sufijo}{extras}.srt"
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

def procesar_en_cola(archivos, motor, num_hablantes, max_palabras, max_segundos, min_palabras, pausa_suave,
                     request: gr.Request, progress=gr.Progress()):
    """Un trabajo por archivo, todos a la cola de una; espera cada uno y devuelve todos los SRT.

    Siempre identifica hablantes: los nombres se ponen después (panel Hablantes o editor).
    """
    archivos = [archivos] if isinstance(archivos, str) else list(archivos or [])
    if not archivos:
        return None
    if motor == "Nemotron" and not nemotron.disponible():
        raise gr.Error("Nemotron no está instalado en este servidor (falta llama-server o el modelo): corré setup.sh.")
    if motor == DEEPINFRA and not nemotron.deepinfra_disponible():
        raise gr.Error("Falta la clave de DeepInfra (DEEPINFRA_API_KEY en ~/.config/whisptest.env).")
    ajustes = _ajustes(max_palabras, max_segundos, min_palabras, pausa_suave)
    cantidad = int(num_hablantes) if str(num_hablantes).isdigit() else None
    trabajos = [cola.encolar("Transcribir", os.path.basename(a), procesar, (a, ajustes, True, cantidad, motor),
                             cola.ip(request)) for a in archivos]
    listos, errores = [], []
    for i, t in enumerate(trabajos, 1):
        prefijo = f"Archivo {i} de {len(trabajos)} · " if len(trabajos) > 1 else ""
        try:
            listos += cola.esperar(t, lambda f, desc="": progress(f, desc=prefijo + desc)) or []
        except gr.Error as e:  # que un archivo que falla no corte la espera de los demás
            errores.append(f"{t.archivo}: {e.message}")
    if errores and not listos:
        raise gr.Error("\n".join(errores))
    if errores:
        gr.Warning("No se pudieron transcribir: " + "; ".join(errores))
    return listos

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
    jsons = [f for f in os.listdir(carpeta) if f.endswith(".json") and not f.endswith((".hablantes.json", ".crudo.json"))]
    jsons.sort(key=lambda f: os.path.getmtime(os.path.join(carpeta, f)), reverse=True)
    return [f[:-len(".json")] + ".srt" for f in jsons]

def _traducciones(nombre_srt):
    """Traducciones de esta transcripción (mismo video y mismo motor), hechas con los loops anteriores."""
    idioma = r"_[A-Za-z]{2,3}(?:-[A-Za-z]{2})?"
    m = re.fullmatch(r"(.*)" + idioma + r"((?:\.(?:whisper|nemotron|deepinfra))?)\.srt", nombre_srt)
    if not m:
        return []
    patron = re.escape(m.group(1)) + idioma + re.escape(m.group(2)) + r"(?:\.nombres)?\.srt"
    propios = {nombre_srt, os.path.basename(srt_nombres(nombre_srt))}
    return sorted(f for f in os.listdir(OUTPUT_DIR) if f not in propios and re.fullmatch(patron, f))

def mostrar_srt(nombre_srt, aviso=""):
    """(archivos, resumen, texto) del SRT elegido; la vista previa es la de nombres si existe.

    Sin SRT, la vista previa y los archivos se ocultan (para no dejar recuadros vacíos).
    """
    archivos = salidas(nombre_srt) if nombre_srt else []
    if not archivos:
        return gr.update(value=None, visible=False), "", gr.update(value="", visible=False)
    with open(archivos[-1], encoding="utf-8") as f:
        texto = f.read()
    return gr.update(value=archivos, visible=True), aviso + loops.resumen_srt(texto), gr.update(value=texto, visible=True)

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
        with gr.Row(equal_height=True):
            input_data = gr.File(
                label="Subí uno o varios videos o audios (.mp3, .wav, .m4a, .mp4, .mkv, etc.): van todos a la cola",
                file_count="multiple",
                file_types=[
                    "video", "audio",
                    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
                    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"
                ],
                type="filepath",
                height=250,
            )
            with gr.Column():
                motor = gr.Radio(list(MOTORES), value="Whisper", label="Motor",
                                 info="Whisper: rápido · Nemotron: mejores hablantes, ~2 min cada 10 de audio · "
                                      "DeepInfra: Nemotron en la nube, ~1 min y ~USD 0,04 cada 20 min. "
                                      "Los Nemotron van en tramos de 5 min y numeran los hablantes de nuevo en cada uno.")
                num_hablantes = gr.Dropdown(["Automático"] + [str(n) for n in range(1, 11)], value="Automático",
                                            label="Cantidad de hablantes",
                                            info="Si la sabés, mejora la precisión. Los nombres se ponen después, "
                                                 "en Hablantes o en el editor.")

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
            with gr.Accordion(f"Hablantes ({len(datos['hablantes'])})", open=False):
                gr.Markdown("Escuchá cada muestra, poné los nombres y aplicalos. "
                            "Si a una persona la partió en dos, poneles el mismo nombre. "
                            f"Para corregir quién dice cada loop mirando el video: "
                            f"[abrir el editor de hablantes ↗](editor?srt={quote(nombre_srt)})")
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

        # Vista previa y archivos aparecen cuando hay una transcripción elegida o una en curso
        with gr.Row(equal_height=True):
            vista = gr.Textbox(label="Vista previa", lines=10, max_lines=10, interactive=False, buttons=["copy"],
                               visible=False, scale=2)
            output_srt = gr.File(label="Archivos SRT (limpio y, si hay hablantes, .nombres.srt)",
                                 file_count="multiple", interactive=False, visible=False, scale=1)

        def al_terminar(paths, n):
            nombre = os.path.basename(paths[0]) if paths else None
            return gr.update(choices=transcripciones(), value=nombre), *mostrar_srt(nombre), n + 1

        btn_transcribir.click(
            lambda: gr.update(visible=True), None, output_srt, show_progress="hidden",  # ahí se ve el avance
        ).then(
            fn=procesar_en_cola,
            inputs=[input_data, motor, num_hablantes, *ajustes],
            outputs=[output_srt],
            concurrency_limit=None,  # la GPU la serializa cola.py
        ).success(al_terminar, [output_srt, refresco], [elegido, output_srt, resumen, vista, refresco])
        btn_rearmar.click(rearmar, [elegido, *ajustes], [output_srt, resumen, vista])
        elegido.input(mostrar_srt, elegido, [output_srt, resumen, vista])
        tab_transcribir.select(lambda: gr.update(choices=transcripciones()), None, elegido, show_progress="hidden")
        btn_por_defecto.click(lambda: valores_def, None, ajustes)
        demo.load(lambda guardados: list(guardados), ajustes_guardados, ajustes)
        gr.on([a.change for a in ajustes], lambda *valores: list(valores), ajustes, ajustes_guardados,
              show_progress="hidden")

    with gr.Tab("Traducir SRT"):
        with gr.Row(equal_height=True):
            input_srt = gr.File(label="Subí tu archivo SRT", file_types=[".srt"], height=250)
            with gr.Column():
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

        with gr.Row(visible=False) as fila_traducido:  # aparece al traducir
            output_traducido = gr.File(label="SRT Traducido", height=120)
            output_estado = gr.Textbox(label="Estado", interactive=False)

        btn_traducir.click(lambda: gr.update(visible=True), None, fila_traducido, show_progress="hidden").then(
            fn=traducir_en_cola,
            inputs=[input_srt, idioma_origen, idioma_destino, instrucciones, modelo_ollama],
            outputs=[output_traducido, output_estado],
            concurrency_limit=None,  # la GPU la serializa cola.py
        )

    return [(btn_transcribir, "Generar SRT de Loops"), (btn_traducir, "Traducir SRT")]
