"""Pestañas Archivos e Historial.

Archivos: los resultados de output_translate y output_separation agrupados por video
(un SRT por idioma y los cuatro audios separados quedan en una sola fila).
Historial: resumen y detalle de uso.log, filtrable por mes y tipo.
"""
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import gradio as gr

import server_separation
import server_translate

LOG_FILE = server_translate.LOG_FILE
DESDE = "2026-09-01"  # antes de esto el log tiene pruebas

AUDIOS = {"_voces_vad": "Habla VAD", "_musica_sfx_vad": "SFX VAD", "_voces": "Habla", "_musica_sfx": "SFX"}
TIPOS = {
    "TRANSCRIPCION": "Transcripción", "TRADUCCION": "Traducción", "SEPARACION": "Separación",
    # nombres que usa cola.py en las líneas ERROR / CANCELADO
    "TRANSCRIBIR": "Transcripción", "TRADUCIR": "Traducción", "SEPARAR": "Separación",
}


def _duracion(segundos):
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _tamano(n):
    for unidad in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unidad}" if unidad == "B" else f"{n:.1f} {unidad}"
        n /= 1024
    return f"{n:.1f} GB"


# ===================== ARCHIVOS =====================

def _clasificar(nombre):
    """(video, etiqueta) para un archivo de salida, o None si no es nuestro."""
    raiz, ext = os.path.splitext(nombre)
    if ext == ".srt":
        m = re.match(r"^(.*)_([A-Za-z]{2,3}(?:-[A-Za-z]{2})?)$", raiz)
        return (m.group(1), f"SRT {m.group(2)}") if m else (raiz, "SRT")
    if ext == ".mp3":
        for sufijo, etiqueta in AUDIOS.items():
            if raiz.endswith(sufijo):
                return raiz[: -len(sufijo)], etiqueta
        return raiz, "Audio"
    return None


def _grupos():
    """{video: {"archivos": [(etiqueta, ruta)], "fecha": mtime, "tamano": bytes}}"""
    grupos = defaultdict(lambda: {"archivos": [], "fecha": 0, "tamano": 0})
    for carpeta in (server_translate.OUTPUT_DIR, server_separation.OUTPUT_DIR):
        if not os.path.isdir(carpeta):
            continue
        for nombre in os.listdir(carpeta):
            ruta = os.path.join(carpeta, nombre)
            clase = _clasificar(nombre) if os.path.isfile(ruta) else None
            if not clase:
                continue
            video, etiqueta = clase
            st = os.stat(ruta)
            g = grupos[video]
            g["archivos"].append((etiqueta, ruta))
            g["fecha"] = max(g["fecha"], st.st_mtime)
            g["tamano"] += st.st_size
    orden = list(AUDIOS.values())
    for g in grupos.values():
        # SRT primero (por idioma), después los audios en el orden de la pestaña Separar
        g["archivos"].sort(key=lambda a: (a[0] in orden, orden.index(a[0]) if a[0] in orden else 0, a[0]))
    return grupos


def _contenido(archivos):
    srts = [e.removeprefix("SRT").strip() or "?" for e, _ in archivos if e.startswith("SRT")]
    audios = [e for e, _ in archivos if not e.startswith("SRT")]
    partes = []
    if srts:
        partes.append("SRT " + ", ".join(srts))
    if audios:
        partes.append(f"{len(audios)} audios" if len(audios) > 1 else audios[0])
    return " · ".join(partes)


def tabla_archivos(buscar="", tipo="Todo"):
    filas = []
    for video, g in sorted(_grupos().items(), key=lambda kv: kv[1]["fecha"], reverse=True):
        etiquetas = [e for e, _ in g["archivos"]]
        if tipo == "Subtítulos" and not any(e.startswith("SRT") for e in etiquetas):
            continue
        if tipo == "Audios" and all(e.startswith("SRT") for e in etiquetas):
            continue
        if buscar and buscar.lower() not in video.lower():
            continue
        fecha = datetime.fromtimestamp(g["fecha"]).strftime("%Y-%m-%d %H:%M")
        filas.append([video, _contenido(g["archivos"]), fecha, _tamano(g["tamano"])])
    return filas


def _vista(ruta):
    """Actualiza reproductor y vista previa según el archivo elegido."""
    if not ruta:
        return gr.update(visible=False, value=None), gr.update(visible=False, value="")
    if ruta.endswith(".srt"):
        with open(ruta, encoding="utf-8", errors="ignore") as f:
            texto = f.read()
        return gr.update(visible=False, value=None), gr.update(visible=True, value=texto)
    return gr.update(visible=True, value=ruta), gr.update(visible=False, value="")


def build_archivos():
    with gr.Tab("Archivos") as tab:
        with gr.Row():
            buscar = gr.Textbox(placeholder="Buscar por nombre...", show_label=False, scale=3)
            tipo = gr.Radio(["Todo", "Subtítulos", "Audios"], value="Todo", show_label=False, scale=2)
            btn_actualizar = gr.Button("🔄 Actualizar", scale=0)

        tabla = gr.Dataframe(
            headers=["Video", "Contenido", "Fecha", "Tamaño"],
            value=tabla_archivos,
            interactive=False,
            wrap=True,
            max_height=420,
            column_widths=["55%", "20%", "15%", "10%"],
        )

        with gr.Group(visible=False) as panel:
            titulo = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    descargas = gr.File(label="Descargar", file_count="multiple", interactive=False)
                with gr.Column(scale=2):
                    elegido = gr.Radio(label="Ver / escuchar", choices=[])
                    reproductor = gr.Audio(type="filepath", interactive=False, visible=False, show_label=False)
                    vista_srt = gr.Textbox(lines=12, max_lines=12, interactive=False, visible=False,
                                           show_label=False, buttons=["copy"])

        def on_select(evt: gr.SelectData):
            video = evt.row_value[0] if evt.row_value else None
            g = _grupos().get(video)
            if not g:
                return gr.update(visible=False), "", None, gr.update(choices=[], value=None)
            opciones = [(etiqueta, ruta) for etiqueta, ruta in g["archivos"]]
            primero = opciones[0][1]
            return (gr.update(visible=True), f"#### {video}", [r for _, r in opciones],
                    gr.update(choices=opciones, value=primero))

        filtros = [buscar, tipo]
        buscar.change(tabla_archivos, filtros, tabla, show_progress="hidden")
        tipo.change(tabla_archivos, filtros, tabla, show_progress="hidden")
        btn_actualizar.click(tabla_archivos, filtros, tabla)
        tab.select(tabla_archivos, filtros, tabla, show_progress="hidden")
        tabla.select(on_select, None, [panel, titulo, descargas, elegido])
        elegido.change(_vista, elegido, [reproductor, vista_srt], show_progress="hidden")


# ===================== HISTORIAL =====================

def _leer_log():
    entradas = []
    if not os.path.exists(LOG_FILE):
        return entradas
    with open(LOG_FILE, encoding="utf-8", errors="ignore") as f:
        for linea in f:
            linea = linea.rstrip("\n")
            partes = linea.split("\t")
            if len(partes) < 3 or partes[0] < DESDE:
                continue
            accion = partes[1]
            campos = dict(re.findall(r"(\w+)=([^\t]*)", linea))
            if accion in ("TRANSCRIPCION", "TRADUCCION", "SEPARACION"):
                tipo, archivo, estado = TIPOS[accion], partes[2], "ok"
                if accion == "TRANSCRIPCION":
                    detalle = f"idioma {partes[3]}" if len(partes) > 3 else ""
                    if campos.get("motor") == "nemotron":
                        detalle += " · Nemotron"
                elif accion == "TRADUCCION":
                    detalle = partes[3].replace("->", " → ") if len(partes) > 3 else ""
                else:
                    detalle = campos.get("modelo", "")
            elif accion in ("ERROR", "CANCELADO") and len(partes) > 3:
                tipo = TIPOS.get(partes[2].upper(), partes[2])
                archivo = partes[3]
                detalle = partes[4] if len(partes) > 4 and "=" not in partes[4] else ""
                estado = accion.lower()
            else:
                continue
            entradas.append({
                "fecha": partes[0][:10], "hora": partes[0][11:16], "tipo": tipo, "archivo": archivo,
                "detalle": detalle, "estado": estado, "ip": campos.get("ip", ""),
                "audio": int(campos["audio"].rstrip("s")) if campos.get("audio", "").rstrip("s").isdigit() else 0,
                "proceso": int(campos["proceso"].rstrip("s")) if campos.get("proceso", "").rstrip("s").isdigit() else 0,
            })
    return entradas


def _meses():
    return ["Todos"] + sorted({e["fecha"][:7] for e in _leer_log()}, reverse=True)


def _tarjeta(titulo, valor, sub=""):
    return (
        '<div style="flex:1;min-width:140px;padding:12px 16px;border-radius:8px;'
        'border:1px solid var(--border-color-primary);background:var(--block-background-fill)">'
        f'<div style="font-size:0.85em;opacity:0.7">{titulo}</div>'
        f'<div style="font-size:1.5em;font-weight:600">{valor}</div>'
        f'<div style="font-size:0.8em;opacity:0.7">{sub}</div></div>'
    )


def historial(mes="Todos", tipo="Todos", buscar=""):
    entradas = [e for e in _leer_log() if mes == "Todos" or e["fecha"].startswith(mes)]
    ok = [e for e in entradas if e["estado"] == "ok"]

    # Tarjetas
    tarjetas = []
    for t in ("Transcripción", "Traducción", "Separación"):
        es = [e for e in ok if e["tipo"] == t]
        audio = sum(e["audio"] for e in es)
        tarjetas.append(_tarjeta(t.replace("ción", "ciones"), len(es), f"{_duracion(audio)} de audio" if audio else ""))
    fallidos = len(entradas) - len(ok)
    tarjetas.append(_tarjeta("Tiempo de proceso", _duracion(sum(e["proceso"] for e in ok)),
                             f"{fallidos} con error o cancelados" if fallidos else ""))
    html = f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(tarjetas)}</div>'

    # Resumen por día (más reciente primero)
    dias = defaultdict(lambda: {"Transcripción": 0, "Traducción": 0, "Separación": 0, "audio": 0, "proceso": 0})
    for e in ok:
        d = dias[e["fecha"]]
        d[e["tipo"]] += 1
        d["audio"] += e["audio"]
        d["proceso"] += e["proceso"]
    resumen = [[dia, v["Transcripción"], v["Traducción"], v["Separación"], _duracion(v["audio"]), _duracion(v["proceso"])]
               for dia, v in sorted(dias.items(), reverse=True)]

    # Detalle
    detalle = []
    for e in reversed(entradas):
        if tipo != "Todos" and e["tipo"] != tipo:
            continue
        if buscar and buscar.lower() not in e["archivo"].lower():
            continue
        detalle.append([
            f'{e["fecha"]} {e["hora"]}', e["tipo"], e["archivo"], e["detalle"],
            _duracion(e["audio"]) if e["audio"] else "", _duracion(e["proceso"]) if e["proceso"] else "",
            "" if e["estado"] == "ok" else e["estado"], e["ip"],
        ])
    return html, resumen, detalle


def build_historial():
    with gr.Tab("Historial") as tab:
        meses = _meses()
        mes_ini = time.strftime("%Y-%m") if time.strftime("%Y-%m") in meses else "Todos"
        with gr.Row():
            mes = gr.Dropdown(meses, value=mes_ini, label="Mes", scale=1)
            tipo = gr.Dropdown(["Todos", "Transcripción", "Traducción", "Separación"], value="Todos",
                               label="Tipo (detalle)", scale=1)
            buscar = gr.Textbox(label="Buscar archivo (detalle)", scale=2)
            btn_actualizar = gr.Button("🔄 Actualizar", scale=0)

        html_ini, resumen_ini, detalle_ini = historial(mes_ini)
        tarjetas = gr.HTML(html_ini)

        with gr.Tabs():
            with gr.Tab("Por día"):
                tabla_resumen = gr.Dataframe(
                    headers=["Día", "Transcripciones", "Traducciones", "Separaciones", "Audio", "Proceso"],
                    value=resumen_ini, interactive=False, max_height=420,
                )
            with gr.Tab("Detalle"):
                tabla_detalle = gr.Dataframe(
                    headers=["Fecha", "Tipo", "Archivo", "Detalle", "Audio", "Proceso", "Estado", "IP"],
                    value=detalle_ini, interactive=False, wrap=True, max_height=500,
                    column_widths=["11%", "9%", "34%", "12%", "7%", "7%", "6%", "14%"],
                )

        filtros = [mes, tipo, buscar]
        salidas = [tarjetas, tabla_resumen, tabla_detalle]
        for comp in (mes, tipo, buscar):
            comp.change(historial, filtros, salidas, show_progress="hidden")
        btn_actualizar.click(historial, filtros, salidas)

        def al_abrir(mes_actual, tipo_actual, buscar_actual):
            return (gr.update(choices=_meses()), *historial(mes_actual, tipo_actual, buscar_actual))

        tab.select(al_abrir, filtros, [mes, *salidas], show_progress="hidden")
