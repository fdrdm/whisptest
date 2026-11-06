"""Transcripción con NVIDIA Nemotron 3.5 Transcribe (GGUF) vía llama-server de llama.cpp.

Alternativa a Whisper. Nemotron devuelve el texto (con etiquetas <spk:N> si se piden
hablantes) pero sin tiempos: server_translate.py los saca alineando ese texto con Whisper.

El servidor se levanta por trabajo y se baja al terminar, como los demás modelos (gpu.py).
Corre con los expertos del MoE en RAM (~10 GB) y el resto en VRAM (~4,6 GB), dentro de un
scope de systemd con tope de RAM: si se pasa, el kernel mata solo al servidor y no se
cuelga la PC. llama-cpp-python todavía no trae el soporte de audio de Nemotron (entró en
llama.cpp el 2026-11-03), por eso se usa el binario compilado (ver setup.sh).

transcribir() solo habla con una API compatible con OpenAI, así que el servidor local se
puede cambiar por uno remoto sin tocar el resto.

deepinfra() usa en cambio el modelo completo (BF16) alojado en DeepInfra, que ya devuelve cada
palabra con sus tiempos y su hablante. Necesita DEEPINFRA_API_KEY en el entorno del servicio
(~/.config/whisptest.env, nunca en el repo).
"""
import base64
import concurrent.futures
import io
import json
import logging
import os
import subprocess
import threading
import time
import wave

import numpy as np
import requests

import gpu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = os.environ.get("NEMOTRON_LLAMA_SERVER", os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
MODELO_DIR = os.environ.get("NEMOTRON_DIR", os.path.expanduser("~/modelos/nemotron-transcribe"))
MODELO = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-IQ4_NL.gguf"
MMPROJ = "mmproj-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-F16.gguf"
PUERTO = 8091
URL = f"http://127.0.0.1:{PUERTO}"
CONTEXTO = 16384       # un tramo de 10 min usa ~7400 de audio + ~2000 de texto (con 65536: +0,3 GB de VRAM)
MAX_SEGUNDOS = 3600    # límite oficial del modelo (por pedido: el local va en tramos de TRAMO_SEG)
TRAMO_SEG = 600        # como DeepInfra: tramos de ~10 min, más estables y sin límite de duración
RAM_MAX = "30G"        # medido: 12,4 GB con 30 min de audio
LOG_SERVIDOR = os.path.join(BASE_DIR, "nemotron.log")
SR = 16000

# Prompts oficiales del model card (con el texto antes del audio)
PROMPT_TEXTO = ("Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; "
                "keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. "
                "Output: transcript only.")
PROMPT_HABLANTES = ("Transcribe this audio with punctuation and capitalization. Include all spoken words and "
                    "verbal fillers. Write numbers in their standard written form. Put speaker tags {tags} on "
                    "the transcription.")

_proceso = None
_lock = threading.Lock()


def disponible():
    return all(os.path.exists(p) for p in (LLAMA_SERVER, os.path.join(MODELO_DIR, MODELO), os.path.join(MODELO_DIR, MMPROJ)))


def _corriendo():
    return _proceso is not None and _proceso.poll() is None


def _iniciar(progress):
    global _proceso
    with _lock:
        if _corriendo():
            return
        logging.info("MODELO\tcargando nemotron...")
        progress(0, desc="Cargando Nemotron...")
        _proceso = subprocess.Popen(
            ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={RAM_MAX}", "-p", "MemorySwapMax=0",
             LLAMA_SERVER, "-m", os.path.join(MODELO_DIR, MODELO), "--mmproj", os.path.join(MODELO_DIR, MMPROJ),
             "--cpu-moe", "-ngl", "99", "-c", str(CONTEXTO), "-np", "1", "--reasoning", "off",
             "--host", "127.0.0.1", "--port", str(PUERTO)],
            stdout=open(LOG_SERVIDOR, "w"), stderr=subprocess.STDOUT,
        )
    limite = time.time() + 180
    while time.time() < limite:
        if not _corriendo():
            _proceso = None
            raise RuntimeError(f"Nemotron no pudo arrancar. Últimas líneas de nemotron.log:\n{_cola_log()}")
        try:
            if requests.get(f"{URL}/health", timeout=2).ok:
                logging.info("MODELO\tnemotron cargado")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    _detener()
    raise RuntimeError("Nemotron tardó más de 3 minutos en cargar.")


def _detener():
    global _proceso
    with _lock:
        if _proceso is None:
            return
        _proceso.terminate()
        try:
            _proceso.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _proceso.kill()
            _proceso.wait()
        _proceso = None
        logging.info("MODELO\tnemotron descargado")


gpu.registrar("nemotron", _detener, _corriendo)


def _cola_log(n=8):
    try:
        with open(LOG_SERVIDOR, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def vram_gb():
    """VRAM que ocupa el servidor (no la cuenta PyTorch), para el log de uso."""
    if not _corriendo():
        return 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return sum(int(m) for p, m in (l.split(",") for l in out.splitlines() if l.strip())
                   if int(p) == _proceso.pid) / 1024
    except Exception:
        return 0.0


def _wav(audio):
    """WAV mono 16 kHz en memoria a partir del audio float32 de whisper.load_audio()."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def tramos(audio, largo=TRAMO_SEG, busqueda=60, ventana=0.5):
    """[(desde, hasta)] en muestras: tramos de ~largo segundos cortados en el medio segundo más
    silencioso del último minuto de cada uno, para no partir palabras."""
    cortes, n = [0], int(ventana * SR)
    while len(audio) - cortes[-1] > largo * SR:
        a = cortes[-1] + (largo - busqueda) * SR
        trozo = audio[a:a + busqueda * SR]
        energia = (trozo[:len(trozo) // n * n].reshape(-1, n) ** 2).mean(1)
        cortes.append(a + int(energia.argmin()) * n + n // 2)
    return list(zip(cortes, cortes[1:] + [len(audio)]))


def prompt(hablantes, cantidad=None):
    if not hablantes:
        return PROMPT_TEXTO
    return PROMPT_HABLANTES.format(tags=" ".join(f"<spk:{i}>" for i in range(cantidad or 4)))


# Penalización de repeticiones de llama.cpp. Solo para reintentar: corta bucles como el "la la la..."
# de una canción (Daria 101), pero también toca repeticiones legítimas ("self-esteem" -> "self esteem")
DRY = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2}


class _Bucle(Exception):
    pass


def _en_bucle(texto, n=150):
    """True si las últimas n palabras son casi todas la misma secuencia repetida."""
    w = texto.lower().split()[-n:]
    if len(w) < n:
        return False
    grupos = [tuple(w[i:i + 6]) for i in range(len(w) - 5)]
    return len(set(grupos)) / len(grupos) < 0.2


def _generar(body, url, esperados, progress, desc):
    partes, fin = [], None
    # Al cancelar, progress() lanza una excepción, se cierra la conexión y el servidor corta la generación
    with requests.post(f"{url}/v1/chat/completions", json=body, stream=True, timeout=(10, 1800)) as r:
        if not r.ok:
            raise RuntimeError(f"Nemotron respondió {r.status_code}: {r.text[:300]}")
        for linea in r.iter_lines():
            if not linea.startswith(b"data: ") or linea == b"data: [DONE]":
                continue
            eleccion = json.loads(linea[6:])["choices"][0]
            if eleccion["delta"].get("content"):
                partes.append(eleccion["delta"]["content"])
                if len(partes) % 20 == 0:
                    progress(min(len(partes) / esperados, 0.99), desc=desc)
                if len(partes) % 100 == 0 and _en_bucle("".join(partes)):
                    raise _Bucle()
            fin = eleccion.get("finish_reason") or fin
    if fin == "length":  # se pasó del tope: casi seguro un bucle que el detector no vio
        raise _Bucle()
    return "".join(partes)


def transcribir(audio, texto_prompt, progress, url=URL):
    """Texto de Nemotron para el audio (float32, 16 kHz). Va mostrando el avance y se puede cancelar."""
    segundos = len(audio) / SR
    if segundos > MAX_SEGUNDOS:
        raise RuntimeError(f"Nemotron acepta hasta {MAX_SEGUNDOS // 60} minutos de audio; este dura "
                           f"{segundos / 60:.0f}. Usá Whisper para este archivo.")
    if url == URL:
        _iniciar(progress)
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": texto_prompt},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(_wav(audio)).decode(), "format": "wav"}},
        ]}],
        "temperature": 0,
        "max_tokens": int(segundos * 6) + 200,  # tope contra bucles (~4 tokens por segundo de habla)
        "stream": True,
    }
    esperados = max(segundos * 4, 50)
    progress(0, desc="Nemotron escuchando el audio...")
    try:
        return _generar(body, url, esperados, progress, "Nemotron transcribiendo...")
    except _Bucle:
        logging.info("NEMOTRON\tse quedó repitiendo: reintento con penalización de repeticiones (DRY)")
    progress(0, desc="Nemotron se quedó repitiendo: reintentando con penalización de repeticiones...")
    try:
        return _generar({**body, **DRY}, url, esperados, progress, "Nemotron transcribiendo (2º intento)...")
    except _Bucle:
        raise RuntimeError("Nemotron se quedó repitiendo aun con penalización de repeticiones. "
                           "Probá con Whisper para este archivo.")


# ===================== DEEPINFRA =====================
# Usa el endpoint propio de DeepInfra (multipart), no el de OpenAI ni OpenRouter: solo ahí se
# puede pedir diarize, que da el hablante de cada palabra.

DEEPINFRA_URL = "https://api.deepinfra.com/v1/inference/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
DEEPINFRA_WHISPER = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"


def deepinfra_disponible():
    return bool(os.environ.get("DEEPINFRA_API_KEY"))


def _ogg(input_file, desde=0, duracion=None):
    """Audio en opus mono 16 kHz (~4 MB cada 20 min): DeepInfra rechaza archivos pesados."""
    tramo = ["-ss", str(desde)] + (["-t", str(duracion)] if duracion else [])
    r = subprocess.run(["ffmpeg", "-loglevel", "error", *tramo, "-i", input_file, "-vn", "-sn", "-dn", "-ac", "1",
                        "-ar", str(SR), "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "pipe:1"],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"No se pudo leer el audio: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout


def deepinfra(input_file, segundos, cantidad, progress):
    """Respuesta de DeepInfra: {"words": [{"word", "start", "end", "speaker"}...], "inference_status": {"cost"...}}.

    Siempre con diarize y la instrucción con <spk:N>: sin eso devuelve una sola voz (y sin
    diarize, en Daria 101 perdió un cuarto de las palabras). Los hablantes vienen numerados
    de nuevo en cada tramo de ~10 min que corta DeepInfra.
    """
    progress(0, desc="Preparando el audio para DeepInfra...")
    audio = _ogg(input_file)
    datos = {"chunk_level": "word", "diarize": "true", "instruction": prompt(True, cantidad)}
    ex = concurrent.futures.ThreadPoolExecutor(1)
    pedido = ex.submit(_pedir, DEEPINFRA_URL, audio, datos, 1800)
    ex.shutdown(wait=False)  # al cancelar no se espera la respuesta (DeepInfra la cobra igual)
    t0 = time.time()
    # No hay avance real: se estima (Daria 101: 114 s para 21 min de audio)
    while not pedido.done():
        s = int(time.time() - t0)
        progress(min(s / (segundos * 0.1 + 30), 0.95), desc=f"Nemotron (DeepInfra) transcribiendo... {s // 60}:{s % 60:02d}")
        time.sleep(1)
    return pedido.result()


def _pedir(url, audio, datos, timeout):
    r = requests.post(url, files={"audio": ("audio.ogg", audio, "audio/ogg")}, data=datos, timeout=timeout,
                      headers={"Authorization": "bearer " + os.environ["DEEPINFRA_API_KEY"]})
    if not r.ok:
        raise RuntimeError(f"DeepInfra respondió {r.status_code}: {r.text[:300]}")
    return r.json()


def deepinfra_idioma(input_file, desde):
    """(idioma, costo) con el Whisper de DeepInfra sobre 30 s de audio desde `desde` (~USD 0,0001)."""
    d = _pedir(DEEPINFRA_WHISPER, _ogg(input_file, desde, 30), {}, 120)
    return d.get("language") or "", d.get("inference_status", {}).get("cost", 0)


def tiempos_sanos(palabras):
    """False si los tiempos se rompieron (pasó en 1 de 6 audios: desde cierto minuto se amontonan
    y se desordenan). Entonces el texto se alinea con Whisper como con el Nemotron local."""
    if not palabras:
        return False
    hacia_atras = sum(b["start"] < a["start"] - 0.5 for a, b in zip(palabras, palabras[1:]))
    inicios = [w["start"] for w in palabras]
    # Nadie dice más de 8 palabras por segundo durante 10 segundos
    amontonadas = any(inicios[i + 80] - inicios[i] < 10 for i in range(len(inicios) - 80))
    return hacia_atras <= len(palabras) / 200 and not amontonadas
