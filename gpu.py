"""Coordinación de VRAM entre Whisper, Demucs y Ollama.

Todo corre en un solo proceso con una única cola de GPU (concurrency_id="gpu"),
así que nunca hay dos trabajos a la vez. Además:
- los modelos propios se cargan por trabajo y se descargan apenas termina
  (cargan en segundos, no vale la pena dejarlos ocupando VRAM);
- PyTorch queda limitado a VRAM_LIMIT_GB para no dejar sin memoria al
  escritorio, a Chrome ni a Ollama;
- antes de empezar se verifica que haya VRAM libre suficiente.
"""
import gc
import os
import subprocess
from contextlib import contextmanager

import gradio as gr
import requests
import torch

GB = 1024 ** 3
VRAM_LIMIT_GB = float(os.environ.get("WHISPTEST_VRAM_GB", "5"))
MARGEN_GB = 1.0  # siempre libre para escritorio y navegador
# Pico estimado por modelo; calibrar con el campo vram_pico de uso.log
NECESARIO_GB = {"whisper": 3.5, "demucs": 3.0}  # medido: whisper 3,5 GB (35 min), demucs_ft 2,4 GB (bloques de 3 min)
OLLAMA_URL = "http://localhost:11434"

_modelos = {}  # nombre -> (descargar, esta_cargado)
_limite_aplicado = False


def registrar(nombre, descargar, esta_cargado):
    _modelos[nombre] = (descargar, esta_cargado)


def _cuda_activa():
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def _aplicar_limite():
    global _limite_aplicado
    if not _limite_aplicado:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_LIMIT_GB * GB / total))
        _limite_aplicado = True


def _liberar_propios(excepto=None):
    for nombre, (descargar, esta_cargado) in _modelos.items():
        if nombre != excepto and esta_cargado():
            descargar()
    gc.collect()
    if _cuda_activa():
        torch.cuda.empty_cache()


def _app_gb():
    return torch.cuda.memory_reserved() / GB if _cuda_activa() else 0.0


def _vram_gpu():
    """(usada, total) en GB de toda la placa, sin crear contexto CUDA."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.splitlines()[0]
    usada, total = (int(x) / 1024 for x in out.split(","))
    return usada, total


def _modelos_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=2)
        return [(m["name"], m.get("size_vram", 0) / GB) for m in r.json().get("models", [])]
    except Exception:
        return []


def _describir_uso():
    partes = [f"Ollama {n} {g:.1f} GB" for n, g in _modelos_ollama() if g > 0]
    if _app_gb() >= 0.1:
        partes.append(f"esta app {_app_gb():.1f} GB")
    return ", ".join(partes) or "escritorio y otros procesos"


@contextmanager
def trabajo(nombre):
    """Prepara la GPU para `nombre` (whisper/demucs) y convierte los OOM en errores claros."""
    if not torch.cuda.is_available():
        yield
        return
    _liberar_propios(excepto=nombre)
    _aplicar_limite()
    usada, total = _vram_gpu()
    disponible = total - usada + _app_gb()
    necesario = NECESARIO_GB[nombre] + MARGEN_GB
    if disponible < necesario:
        raise gr.Error(
            f"No hay VRAM suficiente para {nombre}: hay {disponible:.1f} GB libres y hacen falta "
            f"~{necesario:.1f} GB. En uso: {_describir_uso()}. Esperá a que se libere y reintentá."
        )
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        raise gr.Error(
            f"{nombre} superó el límite de {VRAM_LIMIT_GB:g} GB de VRAM de la app. "
            "Probá con menos shifts o un archivo más corto."
        )
    finally:
        _liberar_propios()


def pico_gb():
    """Pico de VRAM reservada por PyTorch desde el inicio del último trabajo."""
    return torch.cuda.max_memory_reserved() / GB if _cuda_activa() else 0.0


def preparar_ollama():
    """Libera Whisper y Demucs para que el modelo de traducción entre en VRAM."""
    _liberar_propios()


def descargar_ollama(modelo):
    """Saca el modelo de traducción de la VRAM al terminar, sin tocar otros modelos."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def _indicador(color, texto):
    return (
        '<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="width:12px;height:12px;border-radius:50%;background:{color}"></div> {texto}</div>'
    )


def estado_html():
    try:
        usada, total = _vram_gpu()
    except Exception:
        return _indicador("#6b7280", "GPU no disponible")
    cargados = [n.capitalize() for n, (_, esta_cargado) in _modelos.items() if esta_cargado()]
    if cargados:
        app = f"{', '.join(cargados)} ({_app_gb():.1f} de {VRAM_LIMIT_GB:g} GB máx.)"
    else:
        app = "sin modelos cargados"
    ollama = ", ".join(f"{n} ({g:.1f} GB)" for n, g in _modelos_ollama()) or "nada cargado"
    color = "#22c55e" if cargados else "#6b7280"
    return _indicador(color, f"VRAM {usada:.1f}/{total:.1f} GB · App: {app} · Ollama: {ollama}")
