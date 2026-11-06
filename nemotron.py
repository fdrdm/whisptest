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
"""
import base64
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
CONTEXTO = 65536       # el del modelo: alcanza para 1 hora de audio
MAX_SEGUNDOS = 3600    # límite oficial del modelo
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


def prompt(hablantes, cantidad=None):
    if not hablantes:
        return PROMPT_TEXTO
    return PROMPT_HABLANTES.format(tags=" ".join(f"<spk:{i}>" for i in range(cantidad or 4)))


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
    partes, fin, tokens = [], None, 0
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
                tokens += 1
                if tokens % 20 == 0:
                    progress(min(tokens / esperados, 0.99), desc="Nemotron transcribiendo...")
            fin = eleccion.get("finish_reason") or fin
    if fin == "length":
        raise RuntimeError("Nemotron no terminó la transcripción (se quedó repitiendo o se pasó del largo). "
                           "Probá con Whisper para este archivo.")
    return "".join(partes)
