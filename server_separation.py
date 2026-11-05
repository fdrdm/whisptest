import gradio as gr
import torch
import torchaudio
import logging
import time
import os
import gc
import tempfile
import subprocess
import numpy as np
from datetime import datetime

import cola
import gpu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_separation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "uso.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Cargar Silero VAD
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True,
)
get_speech_timestamps = vad_utils[0]

from demucs.pretrained import get_model
from demucs.apply import apply_model
import threading

# Demucs se carga por trabajo; gpu.trabajo() lo descarga al terminar (carga en ~2 s)
_demucs_models = {}
_demucs_lock = threading.Lock()

def _unload_demucs():
    global _demucs_models
    with _demucs_lock:
        if _demucs_models:
            nombres = list(_demucs_models.keys())
            _demucs_models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"MODELO\tdemucs {', '.join(nombres)} descargado")

def get_demucs_model(name):
    with _demucs_lock:
        if name not in _demucs_models:
            # Un solo modelo Demucs en VRAM a la vez
            for otro in list(_demucs_models):
                del _demucs_models[otro]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"MODELO\tcargando demucs {name}...")
            _demucs_models[name] = get_model(name)
            if torch.cuda.is_available():
                _demucs_models[name].cuda()
            logging.info(f"MODELO\tdemucs {name} cargado")
        return _demucs_models[name]

gpu.registrar("demucs", _unload_demucs, lambda: bool(_demucs_models))


def tensor_a_mp3(tensor, sr, path):
    """Guarda un tensor de audio a MP3 via archivo temporal raw."""
    tmp = tempfile.mktemp(suffix=".raw")
    try:
        audio_np = tensor.cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        (audio_np * 32767).clip(-32768, 32767).astype(np.int16).tofile(tmp)
        channels = tensor.shape[0] if tensor.ndim == 2 else 1
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", str(sr), "-ac", str(channels), "-i", tmp,
            "-b:a", "192k", path,
        ], capture_output=True, check=True)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def extraer_audio_wav(input_path, sr=44100):
    """Extrae audio de cualquier formato a tensor stereo via archivo temporal raw."""
    tmp = tempfile.mktemp(suffix=".raw")
    try:
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-ar", str(sr), "-ac", "2", "-f", "s16le", "-y", tmp,
        ], capture_output=True, check=True)
        audio_np = np.fromfile(tmp, dtype=np.int16).astype(np.float32) / 32768.0
        audio_np = audio_np.reshape(-1, 2).T
        return torch.from_numpy(audio_np), sr
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def detectar_habla(wav_tensor, sr, threshold):
    """Detecta segmentos con habla usando Silero VAD."""
    if sr != 16000:
        wav_16k = torchaudio.functional.resample(wav_tensor, sr, 16000)
    else:
        wav_16k = wav_tensor
    if wav_16k.shape[0] > 1:
        wav_16k = wav_16k.mean(dim=0, keepdim=True)

    timestamps = get_speech_timestamps(
        wav_16k.squeeze(),
        vad_model,
        threshold=threshold,
        speech_pad_ms=100,
        sampling_rate=16000,
        return_seconds=True,
    )
    return timestamps


def separar_demucs_chunked(wav, model, shifts, overlap, sr=44100,
                           chunk_min=3, overlap_sec=15, progress_fn=None):
    """Procesa Demucs en chunks para no reventar la memoria con audios largos."""
    chunk_samples = int(chunk_min * 60 * sr)
    overlap_samples = int(overlap_sec * sr)
    total_samples = wav.shape[1]

    vocals_idx = model.sources.index("vocals")
    vocals = torch.zeros_like(wav)
    no_vocals = torch.zeros_like(wav)

    # Calcular posiciones de inicio de cada chunk
    starts = []
    pos = 0
    while pos < total_samples:
        starts.append(pos)
        next_pos = pos + chunk_samples
        if next_pos >= total_samples:
            break
        pos = next_pos - overlap_samples

    t_demucs = time.time()
    for i, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)

        if progress_fn:
            elapsed = time.time() - t_demucs
            elapsed_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
            progress_fn(0.1 + 0.75 * (i / len(starts)),
                        desc=f"Demucs chunk {i+1}/{len(starts)} ({elapsed_str})")

        chunk = wav[:, start:end].unsqueeze(0)
        if torch.cuda.is_available():
            chunk = chunk.cuda()

        with torch.no_grad():
            sources = apply_model(model, chunk, shifts=int(shifts),
                                  overlap=overlap, progress=False)

        chunk_vocals = sources[0, vocals_idx].cpu()
        chunk_no_vocals = torch.zeros_like(chunk_vocals)
        for j in range(sources.shape[1]):
            if j != vocals_idx:
                chunk_no_vocals += sources[0, j].cpu()

        del sources, chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chunk_len = end - start
        if i > 0 and overlap_samples > 0:
            fade_len = min(overlap_samples, chunk_len)
            fade = torch.linspace(0, 1, fade_len).unsqueeze(0)
            vocals[:, start:start+fade_len] = (
                vocals[:, start:start+fade_len] * (1 - fade) +
                chunk_vocals[:, :fade_len] * fade
            )
            no_vocals[:, start:start+fade_len] = (
                no_vocals[:, start:start+fade_len] * (1 - fade) +
                chunk_no_vocals[:, :fade_len] * fade
            )
            if fade_len < chunk_len:
                vocals[:, start+fade_len:end] = chunk_vocals[:, fade_len:]
                no_vocals[:, start+fade_len:end] = chunk_no_vocals[:, fade_len:]
        else:
            vocals[:, start:end] = chunk_vocals
            no_vocals[:, start:end] = chunk_no_vocals

    return vocals, no_vocals


def separar(input_file, modelo_demucs_name, demucs_shifts, demucs_overlap,
            vad_threshold, *, ip, progress):
    if not input_file:
        return None, None, None, None, ""

    t0 = time.time()
    nombre_base = os.path.splitext(os.path.basename(input_file))[0]

    # 1. Extraer audio
    progress(0, desc="Extrayendo audio...")
    wav, sr = extraer_audio_wav(input_file, sr=44100)
    duracion_audio = wav.shape[1] / sr

    # 2. Demucs en chunks
    with gpu.trabajo("demucs"):
        model = get_demucs_model(modelo_demucs_name)
        vocals, no_vocals = separar_demucs_chunked(
            wav, model, shifts=demucs_shifts, overlap=demucs_overlap,
            sr=sr, progress_fn=progress,
        )

    # 3. VAD sobre las voces separadas → máscara
    progress(0.85, desc="Aplicando VAD...")
    timestamps = detectar_habla(vocals, sr, vad_threshold)

    fade_samples = int(0.050 * sr)  # 50ms fade
    mask = torch.zeros(1, vocals.shape[1])
    for ts in timestamps:
        start_sample = int(ts['start'] * sr)
        end_sample = min(int(ts['end'] * sr), vocals.shape[1])
        mask[0, start_sample:end_sample] = 1.0

        # Fade en el lado sin voz para no cortar de golpe
        fade_in_start = max(0, start_sample - fade_samples)
        if fade_in_start < start_sample:
            n = start_sample - fade_in_start
            mask[0, fade_in_start:start_sample] = torch.linspace(0, 1, n)

        fade_out_end = min(vocals.shape[1], end_sample + fade_samples)
        if end_sample < fade_out_end:
            n = fade_out_end - end_sample
            mask[0, end_sample:fade_out_end] = torch.linspace(1, 0, n)

    # habla_vad: voces Demucs donde hay habla, silencio donde no
    # sfx_vad: SFX Demucs donde hay habla, audio original donde no
    vocals_vad = vocals * mask
    no_vocals_vad = no_vocals * mask + wav * (1 - mask)

    # 5. Exportar
    progress(0.92, desc="Exportando MP3...")
    vocals_vad_mp3 = os.path.join(OUTPUT_DIR, f"{nombre_base}_voces_vad.mp3")
    sfx_vad_mp3 = os.path.join(OUTPUT_DIR, f"{nombre_base}_musica_sfx_vad.mp3")
    vocals_mp3 = os.path.join(OUTPUT_DIR, f"{nombre_base}_voces.mp3")
    sfx_mp3 = os.path.join(OUTPUT_DIR, f"{nombre_base}_musica_sfx.mp3")

    tensor_a_mp3(vocals_vad, sr, vocals_vad_mp3)
    tensor_a_mp3(no_vocals_vad, sr, sfx_vad_mp3)
    tensor_a_mp3(vocals, sr, vocals_mp3)
    tensor_a_mp3(no_vocals, sr, sfx_mp3)

    duracion_proceso = time.time() - t0
    duracion_habla = sum(ts['end'] - ts['start'] for ts in timestamps)
    logging.info(
        f"SEPARACION\t{nombre_base}\tmodelo={modelo_demucs_name}\t"
        f"audio={duracion_audio:.0f}s\thabla={duracion_habla:.0f}s\tproceso={duracion_proceso:.0f}s\t"
        f"segmentos_habla={len(timestamps)}\tvram_pico={gpu.pico_gb():.1f}GB\tip={ip}"
    )

    progress(1.0, desc="Listo!")
    estado = f"Listo en {duracion_proceso:.0f}s | Audio: {duracion_audio/60:.1f} min | {modelo_demucs_name} shifts={int(demucs_shifts)} | {len(timestamps)} segmentos de habla"

    return vocals_vad_mp3, sfx_vad_mp3, vocals_mp3, sfx_mp3, estado


def separar_en_cola(input_file, modelo_demucs_name, demucs_shifts, demucs_overlap,
                    vad_threshold, request: gr.Request, progress=gr.Progress()):
    if not input_file:
        return None, None, None, None, ""
    t = cola.encolar("Separar", os.path.basename(input_file), separar,
                     (input_file, modelo_demucs_name, demucs_shifts, demucs_overlap, vad_threshold),
                     cola.ip(request))
    return cola.esperar(t, progress)


def format_size(bytes_size):
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f} MB"

def listar_archivos_output():
    if not os.path.exists(OUTPUT_DIR):
        return [], []
    archivos = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
    archivos.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    tabla = []
    for f in archivos:
        path = os.path.join(OUTPUT_DIR, f)
        stat = os.stat(path)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size = format_size(stat.st_size)
        tabla.append([f, size, mtime])
    return archivos, tabla

# ===================== INTERFAZ =====================
# app.py arma la página unificada con estas piezas.

def build_tabs():
    with gr.Tab("Separar audio"):
        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Subí tu Video o Audio (.mp3, .wav, .m4a, .aac, .flac, .ogg, .mp4, .mkv, etc.)",
                    file_types=[
                        "video", "audio",
                        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
                        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"
                    ],
                    type="filepath",
                )

            with gr.Column(scale=1):
                modelo_demucs = gr.Dropdown(
                    choices=["htdemucs_ft", "htdemucs", "mdx_extra"],
                    value="htdemucs_ft",
                    label="Modelo Demucs",
                    info="htdemucs_ft es el mas preciso, htdemucs es mas rapido",
                )

                vad_threshold = gr.Slider(0.1, 0.95, value=0.5, step=0.05,
                                          label="VAD Threshold",
                                          info="Bajo = agarra mas habla. Alto = mas estricto")

                with gr.Accordion("Config Demucs", open=False):
                    demucs_shifts = gr.Slider(0, 10, value=5, step=1,
                                              label="Shifts",
                                              info="Mas = mejor calidad pero mas lento. 0=rapido, 5=muy bueno")
                    demucs_overlap = gr.Slider(0.0, 0.5, value=0.25, step=0.05,
                                               label="Overlap",
                                               info="Solapamiento interno de Demucs (0.25 = default)")

        btn = gr.Button("Separar", variant="primary", size="lg")

        gr.Markdown("### Con VAD (original donde no hay habla)")
        with gr.Row():
            out_voces_vad = gr.Audio(label="Habla VAD", type="filepath")
            out_musica_vad = gr.Audio(label="SFX VAD", type="filepath")

        gr.Markdown("### Sin VAD (Demucs puro)")
        with gr.Row():
            out_voces = gr.Audio(label="Habla", type="filepath")
            out_musica = gr.Audio(label="SFX", type="filepath")

        out_estado = gr.Textbox(label="Estado", interactive=False)

        btn.click(
            fn=separar_en_cola,
            inputs=[input_file, modelo_demucs, demucs_shifts, demucs_overlap, vad_threshold],
            outputs=[out_voces_vad, out_musica_vad, out_voces, out_musica, out_estado],
            concurrency_limit=None,  # la GPU la serializa cola.py
        )


def build_archivos():
    gr.Markdown("### 🎧 Audios separados (`output_separation`)")
    gr.Markdown("Selecciona o reproduce cualquier archivo MP3 generado previamente:")

    archivos_init, tabla_archivos_init = listar_archivos_output()
    primer_archivo_path = os.path.join(OUTPUT_DIR, archivos_init[0]) if archivos_init else None

    with gr.Row():
        selector_archivo = gr.Dropdown(
            choices=archivos_init,
            value=archivos_init[0] if archivos_init else None,
            label="Seleccionar archivo de audio",
            scale=3,
        )
        btn_refrescar_archivos = gr.Button("🔄 Refrescar Archivos", scale=1)

    with gr.Row():
        file_descarga = gr.File(
            label="Descargar archivo",
            value=primer_archivo_path,
            interactive=False,
        )
        audio_preview = gr.Audio(
            label="Reproductor de Audio",
            value=primer_archivo_path,
            type="filepath",
            interactive=False,
        )

    tabla_archivos = gr.Dataframe(
        headers=["Nombre de Archivo", "Tamaño", "Fecha de Modificación"],
        value=tabla_archivos_init,
        interactive=False,
        wrap=True,
        max_height=250,
        label="Archivos disponibles en output_separation (haz clic para seleccionarlo)",
    )

    def on_seleccionar_archivo(archivo=None):
        if archivo:
            path = os.path.join(OUTPUT_DIR, archivo)
            if os.path.exists(path):
                return path, path
        return None, None

    def on_refrescar_archivos():
        archs, tbl = listar_archivos_output()
        primer = os.path.join(OUTPUT_DIR, archs[0]) if archs else None
        return gr.update(choices=archs, value=archs[0] if archs else None), tbl, primer, primer

    def on_select_tabla(evt: gr.SelectData = None):
        if evt is None:
            return gr.update(), None, None
        row_idx = evt.index[0]
        archs, _ = listar_archivos_output()
        if 0 <= row_idx < len(archs):
            arch = archs[row_idx]
            path = os.path.join(OUTPUT_DIR, arch)
            return arch, path, path
        return gr.update(), None, None

    selector_archivo.change(
        fn=on_seleccionar_archivo,
        inputs=[selector_archivo],
        outputs=[file_descarga, audio_preview],
    )
    btn_refrescar_archivos.click(
        fn=on_refrescar_archivos,
        outputs=[selector_archivo, tabla_archivos, file_descarga, audio_preview],
    )
    tabla_archivos.select(
        fn=on_select_tabla,
        outputs=[selector_archivo, file_descarga, audio_preview],
    )
