"""Identificación de hablantes con pyannote (speaker-diarization-community-1).

Corre después de Whisper, dentro del mismo trabajo de la cola. Cada palabra queda
asignada a un hablante y se guarda en palabras/<video>_<idioma>.hablantes.json,
junto con una muestra de audio de cada uno para poder ponerles nombre.
"""
import json
import logging
import os
import threading
import wave
from collections import defaultdict

import numpy as np
import torch

import gpu

MODELO = "pyannote/speaker-diarization-community-1"
LOTE = 8  # medido: 0,6 GB de VRAM con lotes de 8 (2 GB con 32), misma velocidad y resultado
MUESTRA_SEG = 8.0
TRAMO_MINIMO_SEG = 0.5  # tramos más cortos (o de una palabra) se pegan al hablante vecino
SR = 16000

_pipeline = None
_lock = threading.Lock()


def _get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            from pyannote.audio import Pipeline  # tarda en importar; solo si se usa
            logging.info("MODELO\tcargando pyannote...")
            _pipeline = Pipeline.from_pretrained(MODELO).to(torch.device("cuda"))
            _pipeline.segmentation_batch_size = _pipeline.embedding_batch_size = LOTE
            logging.info("MODELO\tpyannote cargado")
        return _pipeline


def _unload():
    global _pipeline
    with _lock:
        if _pipeline is not None:
            _pipeline = None
            logging.info("MODELO\tpyannote descargado")


gpu.registrar("pyannote", _unload, lambda: _pipeline is not None)


def diarizar(audio, num_hablantes=None, progress=None):
    """Turnos (inicio, fin, etiqueta) sin superposición para un audio mono de 16 kHz."""
    pasos = {"segmentation": 0.2, "speaker_counting": 0.3, "embeddings": 0.9, "discrete_diarization": 1.0}

    def hook(paso, _artefacto, file=None, total=None, completed=None):
        if progress:
            fraccion = completed / total if total and completed is not None else 1.0
            anterior = max([v for k, v in pasos.items() if v < pasos.get(paso, 1.0)], default=0.0)
            progress(anterior + (pasos.get(paso, 1.0) - anterior) * fraccion, desc="Identificando hablantes...")

    with gpu.trabajo("pyannote"):
        pipeline = _get_pipeline()
        salida = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR},
                          num_speakers=num_hablantes or None, hook=hook)
        del pipeline
    return [(t.start, t.end, h) for t, _, h in salida.exclusive_speaker_diarization.itertracks(yield_label=True)]


def asignar(palabras, turnos):
    """Hablante de cada palabra: el turno que más se superpone, o el más cercano si no hay."""
    resultado = []
    j = 0  # turnos y palabras vienen ordenados: se recorren juntos
    for w in palabras:
        while j < len(turnos) and turnos[j][1] <= w.start:
            j += 1
        mejor, mejor_sup = None, 0.0
        k = j
        while k < len(turnos) and turnos[k][0] < w.end:
            sup = min(w.end, turnos[k][1]) - max(w.start, turnos[k][0])
            if sup > mejor_sup:
                mejor, mejor_sup = turnos[k][2], sup
            k += 1
        if mejor is None and turnos:
            # Palabra en un silencio: el turno que terminó antes o el que empieza después, el más cercano
            antes = turnos[j - 1] if j > 0 else None
            despues = turnos[j] if j < len(turnos) else None
            if antes and (not despues or w.start - antes[1] <= despues[0] - w.end):
                mejor = antes[2]
            else:
                mejor = despues[2]
        resultado.append(mejor)
    resultado = _suavizar(palabras, resultado)
    # Renombrar por orden de aparición: H1, H2...
    orden = {}
    for h in resultado:
        if h is not None and h not in orden:
            orden[h] = f"H{len(orden) + 1}"
    return [orden.get(h, "H1") for h in resultado]


def _suavizar(palabras, etiquetas):
    """pyannote salta de hablante en palabras sueltas ("You guys | are | related.").

    Un tramo de una sola palabra o de menos de TRAMO_MINIMO_SEG pasa al vecino: al que
    lo rodea si ambos son el mismo, si no al más cercano en el tiempo.
    """
    etiquetas = list(etiquetas)
    for _ in range(3):
        tramos = []  # [inicio, fin) en índices de palabras
        for i, h in enumerate(etiquetas):
            if tramos and etiquetas[tramos[-1][0]] == h:
                tramos[-1][1] = i + 1
            else:
                tramos.append([i, i + 1])
        cambios = False
        for n, (a, b) in enumerate(tramos):
            corto = b - a == 1 or palabras[b - 1].end - palabras[a].start < TRAMO_MINIMO_SEG
            if not corto or len(tramos) == 1:
                continue
            previo = tramos[n - 1] if n > 0 else None
            siguiente = tramos[n + 1] if n + 1 < len(tramos) else None
            hueco_previo = palabras[a].start - palabras[previo[1] - 1].end if previo else float("inf")
            hueco_siguiente = palabras[siguiente[0]].start - palabras[b - 1].end if siguiente else float("inf")
            vecino = previo if hueco_previo <= hueco_siguiente else siguiente
            nuevo = etiquetas[vecino[0]]
            if nuevo != etiquetas[a]:
                etiquetas[a:b] = [nuevo] * (b - a)
                cambios = True
        if not cambios:
            break
    return etiquetas


def _guardar_wav(audio, path):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def guardar(path_json, palabras, etiquetas, audio):
    """Guarda hablante por palabra, estadísticas y una muestra de audio de cada uno."""
    carpeta = os.path.splitext(path_json)[0]  # .../video_en.hablantes/
    os.makedirs(carpeta, exist_ok=True)
    segundos = defaultdict(float)
    tramos = defaultdict(list)  # hablante -> [(inicio, fin, texto)] de palabras seguidas
    for w, h in zip(palabras, etiquetas):
        segundos[h] += w.end - w.start
        if tramos[h] and tramos[h][-1][1] >= w.start - 0.5:
            inicio, _, texto = tramos[h][-1]
            tramos[h][-1] = (inicio, w.end, texto + w.word)
        else:
            tramos[h].append((w.start, w.end, w.word))
    hablantes = {}
    for h in sorted(segundos, key=lambda x: int(x[1:])):
        inicio, fin, _ = max(tramos[h], key=lambda t: min(t[1] - t[0], MUESTRA_SEG))
        fin = min(fin, inicio + MUESTRA_SEG)
        muestra = os.path.join(carpeta, f"{h}.wav")
        _guardar_wav(audio[int(inicio * SR):int(fin * SR)], muestra)
        _, _, frase = max(tramos[h], key=lambda t: min(t[1] - t[0], MUESTRA_SEG))
        hablantes[h] = {"nombre": "", "segundos": round(segundos[h], 1), "frase": frase.strip(), "muestra": muestra}
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"palabras": etiquetas, "hablantes": hablantes}, f, ensure_ascii=False, indent=1)
    return hablantes


def cargar(path_json):
    if not os.path.exists(path_json):
        return None
    with open(path_json, encoding="utf-8") as f:
        return json.load(f)
