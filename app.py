"""Whisptest unificado (ia.fdrdm.com): transcripción, traducción y separación en un solo proceso.

Todos los trabajos de GPU comparten la cola "gpu" y pasan por gpu.py, que limita
la VRAM de la app (WHISPTEST_VRAM_GB, 5 GB por defecto) y mantiene un solo modelo cargado.
"""
import gradio as gr

import gpu
import server_separation
import server_translate
from chunk_uploader import launch_server

with gr.Blocks(title="IA fdrdm") as demo:
    gr.Markdown("# Subtítulos, Traducción y Separación de Audio 🎬🌐🎧")
    indicador_gpu = gr.HTML(value=gpu.estado_html)

    server_translate.build_tabs()
    server_separation.build_tabs()

    with gr.Tab("Archivos e Historial"):
        server_translate.build_archivos()
        gr.Markdown("---")
        server_separation.build_archivos()
        gr.Markdown("---")
        server_translate.build_historial()

    timer = gr.Timer(5)
    timer.tick(fn=gpu.estado_html, outputs=[indicador_gpu])

if __name__ == "__main__":
    launch_server(demo, port=8188, allowed_paths=[server_translate.OUTPUT_DIR, server_separation.OUTPUT_DIR])
