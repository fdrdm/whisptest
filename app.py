"""Whisptest unificado (ia.fdrdm.com): transcripción, traducción y separación en un solo proceso.

Todos los trabajos van a una sola cola (cola.py) que los procesa de a uno, y pasan por
gpu.py, que limita la VRAM de la app (WHISPTEST_VRAM_GB, 5 GB por defecto).
"""
import os

import gradio as gr

import archivos
import cola
import editor
import gpu
import server_separation
import server_translate
from chunk_uploader import launch_server


def estado():
    return f'<div class="estado">{cola.resumen_html()}<div class="estado-gpu">{gpu.estado_html()}</div></div>'


# Verde de wider-media.com (--brand-primary #038169) sobre negro
VERDE = gr.themes.Color(
    c50="#e6f4f0", c100="#c2e6dc", c200="#9ad7c5", c300="#5fbfa5", c400="#04a487", c500="#038169",
    c600="#036c58", c700="#025a49", c800="#023d32", c900="#01241d", c950="#011712", name="wider",
)


def tema():
    t = gr.themes.Base(
        primary_hue=VERDE, secondary_hue=VERDE, neutral_hue=gr.themes.colors.neutral,
        font=["ui-sans-serif", "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        # Tarjetas tipo bento: bordes redondeados y la misma separación entre todo
        block_radius="16px",
        container_radius="16px",
        layout_gap="14px",
        body_background_fill_dark="#0a0a0a",
        background_fill_primary_dark="#141414",
        background_fill_secondary_dark="#0f0f0f",
        block_background_fill_dark="#141414",
        border_color_primary_dark="#262626",
        button_primary_background_fill_dark="#038169",
        button_primary_background_fill_hover_dark="#04a487",
        button_primary_text_color_dark="#ffffff",
        color_accent_soft_dark="#023d32",
        link_text_color_dark="#04a487",
        slider_color_dark="#038169",
        checkbox_background_color_selected_dark="#038169",
    )
    # Siempre oscuro: los valores del modo claro pasan a ser los del oscuro (None = igual que el claro)
    for nombre in dir(t):
        if nombre.endswith("_dark") and hasattr(t, nombre[:-len("_dark")]) and getattr(t, nombre) is not None:
            setattr(t, nombre[:-len("_dark")], getattr(t, nombre))
    return t


with gr.Blocks(title="IA fdrdm") as demo:
    with gr.Row(equal_height=True, elem_id="encabezado"):
        gr.Markdown("# Subtítulos, Traducción y Separación de Audio 🎬🌐🎧")
        indicador = gr.HTML(value=estado)

    # (botón, texto normal): si hay trabajos en la cola pasan a decir "Agregar a la cola"
    botones = server_translate.build_tabs(demo) + server_separation.build_tabs()
    cola.build_tab()
    archivos.build_archivos()
    archivos.build_historial()

    def refrescar():
        return estado(), *[gr.update(value=cola.etiqueta_boton(texto)) for _, texto in botones]

    timer = gr.Timer(3)
    timer.tick(fn=refrescar, outputs=[indicador, *[b for b, _ in botones]], show_progress="hidden")

if __name__ == "__main__":
    launch_server(demo, port=8188, allowed_paths=[server_translate.OUTPUT_DIR, server_separation.OUTPUT_DIR],
                  theme=tema(), favicon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "favicon.png"),
                  routers=[editor.router])
