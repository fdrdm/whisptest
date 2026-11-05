"""Whisptest unificado (ia.fdrdm.com): transcripción, traducción y separación en un solo proceso.

Todos los trabajos van a una sola cola (cola.py) que los procesa de a uno, y pasan por
gpu.py, que limita la VRAM de la app (WHISPTEST_VRAM_GB, 5 GB por defecto).
"""
import gradio as gr

import archivos
import cola
import gpu
import server_separation
import server_translate
from chunk_uploader import launch_server


def estado():
    return gpu.estado_html() + cola.resumen_html()


with gr.Blocks(title="IA fdrdm") as demo:
    gr.Markdown("# Subtítulos, Traducción y Separación de Audio 🎬🌐🎧")
    indicador_gpu = gr.HTML(value=estado)

    server_translate.build_tabs()
    server_separation.build_tabs()
    cola.build_tab()
    archivos.build_archivos()
    archivos.build_historial()

    timer = gr.Timer(5)
    timer.tick(fn=estado, outputs=[indicador_gpu])

if __name__ == "__main__":
    launch_server(demo, port=8188, allowed_paths=[server_translate.OUTPUT_DIR, server_separation.OUTPUT_DIR])
