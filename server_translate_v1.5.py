import gradio as gr
import stable_whisper
import requests
import re
import os
import tempfile

model = stable_whisper.load_model('turbo')

OLLAMA_URL = "https://ollama.fdrdm.com/api/generate"
OLLAMA_MODEL = "translategemma:latest"

PROMPT_TEMPLATE = """You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator for subtitle blocks.
{instrucciones_extra}
The text is split into numbered subtitle blocks. Translate each one keeping the same numbering format. Use context from surrounding blocks to choose the best translation for ambiguous words.
Reply ONLY with the translated blocks in this exact format:
[1] translated text
[2] translated text
...

{text}"""

BATCH_SIZE = 10

IDIOMAS = {
    "English": "en",
    "Spanish": "es",
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

# "Cortar antes de" = preposiciones y conjunciones subordinantes (inicio de clausula)
# NO incluye and/or/but/so/yet/nor porque esas van en "No dejar al final"
DEFAULTS_CORTAR_ANTES = (
    "because, although, though, while, when, if, since, unless, until, whether, "
    "however, moreover, nevertheless, otherwise, whereas, meanwhile, "
    "with, from, about, into, through, during, before, after, "
    "between, against, without, beyond, despite, albeit"
)

# "No dejar al final" = conjunciones coordinantes que no deben quedar colgadas
# al final de un bloque (se mueven al siguiente).
# Tambien funcionan como puntos de corte automaticamente.
DEFAULTS_NO_DEJAR_AL_FINAL = "and, or, but, so, yet, nor"

# Puntuacion que marca fin de oracion o punto de corte
DEFAULTS_PUNTUACION = ". ? ! ,"


def segundos_a_srt(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos - int(segundos)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def transcribir(video_path, max_dur, max_pal, min_pal, min_pal_suave,
                pausa_min, tolerancia, cortar_antes_str, no_dejar_str, puntuacion_str,
                progress=gr.Progress(track_tqdm=True)):
    if not video_path:
        return [], -1, "", gr.update(choices=[], value=None), 0, 0, 0, "", "**Segmento --**"

    result = model.transcribe(video_path)

    todas_las_palabras = []
    for segment in result.segments:
        for word in segment.words:
            todas_las_palabras.append(word)

    MIN_PALABRAS = int(min_pal)
    MIN_PALABRAS_SUAVE = int(min_pal_suave)
    MAX_PALABRAS = int(max_pal)
    MAX_DURACION = float(max_dur)
    TOLERANCIA_ADELANTE = int(tolerancia)
    PAUSA_MINIMA_SUAVE = float(pausa_min)

    CONJUNCIONES = {w.strip().lower() for w in no_dejar_str.split(',') if w.strip()}
    # CORTE_SUAVE = "cortar antes de" + conjunciones (las conjunciones tambien son puntos de corte)
    CORTE_SUAVE = {w.strip().lower() for w in cortar_antes_str.split(',') if w.strip()}
    CORTE_SUAVE |= CONJUNCIONES

    PUNTUACION = tuple(p.strip() for p in puntuacion_str.split() if p.strip())

    loops = []
    loop_actual = []
    total_palabras = len(todas_las_palabras)
    i = 0

    while i < total_palabras:
        word = todas_las_palabras[i]
        palabra = word.word.strip().lower().rstrip('.,!?:;')

        if loop_actual and len(loop_actual) >= 2:
            ultima = loop_actual[-1].word.strip()
            primera_letra = word.word.strip()[:1]
            if ultima.endswith(PUNTUACION) and primera_letra.isupper():
                loops.append(loop_actual)
                loop_actual = []

        if loop_actual and len(loop_actual) >= MIN_PALABRAS_SUAVE and palabra in CORTE_SUAVE:
            pausa = word.start - loop_actual[-1].end
            if pausa >= PAUSA_MINIMA_SUAVE:
                ultima_palabra = loop_actual[-1].word.strip().lower().rstrip('.,!?:;')
                if ultima_palabra in CONJUNCIONES and len(loop_actual) > 1:
                    colgada = loop_actual.pop()
                    loops.append(loop_actual)
                    loop_actual = [colgada]
                else:
                    loops.append(loop_actual)
                    loop_actual = []

        loop_actual.append(word)

        cantidad_palabras = len(loop_actual)
        duracion = loop_actual[-1].end - loop_actual[0].start
        termina_oracion = word.word.strip().endswith(PUNTUACION)
        es_ultima = (i == total_palabras - 1)

        if es_ultima:
            loops.append(loop_actual)
            break

        cortar = False

        if termina_oracion and cantidad_palabras >= MIN_PALABRAS:
            cortar = True

        if not cortar and (cantidad_palabras >= MAX_PALABRAS or duracion >= MAX_DURACION):
            corte_atras = -1
            for j in range(len(loop_actual) - 1, MIN_PALABRAS - 1, -1):
                if loop_actual[j].word.strip().endswith(PUNTUACION):
                    corte_atras = j
                    break

            if corte_atras >= 0:
                loops.append(loop_actual[:corte_atras + 1])
                loop_actual = loop_actual[corte_atras + 1:]
                i += 1
                continue
            else:
                palabras_extra = 0
                for j in range(1, TOLERANCIA_ADELANTE + 1):
                    if i + j >= total_palabras:
                        break
                    w_adelante = todas_las_palabras[i + j]
                    if w_adelante.word.strip().endswith(PUNTUACION):
                        for k in range(1, j + 1):
                            loop_actual.append(todas_las_palabras[i + k])
                        palabras_extra = j
                        break
                i += palabras_extra
                cortar = True

        if cortar:
            loops.append(loop_actual)
            loop_actual = []

        i += 1

    segments = []
    for loop in loops:
        segments.append({
            'start': round(loop[0].start, 3),
            'end': round(loop[-1].end, 3),
            'text': ''.join(w.word for w in loop).strip()
        })

    choices = [f"Segmento {i+1}" for i in range(len(segments))]

    if segments:
        seg = segments[0]
        return (
            segments, 0, video_path,
            gr.update(choices=choices, value="Segmento 1"),
            round(seg['start'], 2), round(seg['end'], 2),
            round(seg['end'] - seg['start'], 2),
            seg['text'], f"**Segmento 1 / {len(segments)}**",
        )

    return segments, -1, video_path, gr.update(choices=[], value=None), 0, 0, 0, "", "**Segmento --**"


def on_select_segment(choice, segments):
    if not choice or not segments:
        return -1, 0, 0, 0, "", "**Segmento --**"

    idx = int(choice.split()[-1]) - 1
    if idx < 0 or idx >= len(segments):
        return -1, 0, 0, 0, "", "**Segmento --**"

    seg = segments[idx]
    dur = round(seg['end'] - seg['start'], 2)
    return (
        idx,
        round(seg['start'], 2), round(seg['end'], 2), dur,
        seg['text'], f"**Segmento {idx + 1} / {len(segments)}**",
    )


def on_update_texto(texto, segments, selected_idx):
    if 0 <= selected_idx < len(segments):
        segments[selected_idx]['text'] = texto
    return segments


def on_update_inicio(inicio, segments, selected_idx):
    if 0 <= selected_idx < len(segments):
        segments[selected_idx]['start'] = round(inicio, 3)
        dur = round(segments[selected_idx]['end'] - inicio, 2)
    else:
        dur = 0
    return segments, dur


def on_update_fin(fin, segments, selected_idx):
    if 0 <= selected_idx < len(segments):
        segments[selected_idx]['end'] = round(fin, 3)
        dur = round(fin - segments[selected_idx]['start'], 2)
    else:
        dur = 0
    return segments, dur


def guardar_srt(segments, video_path):
    if not segments:
        return None
    nombre = os.path.splitext(os.path.basename(video_path))[0] if video_path else "subtitulos"
    output_dir = tempfile.mkdtemp()
    path = os.path.join(output_dir, f"{nombre}.srt")
    with open(path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            f.write(f"{i+1}\n")
            f.write(f"{segundos_a_srt(seg['start'])} --> {segundos_a_srt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    return path


def nav_segmento(direction, segments, selected_idx):
    if not segments:
        return gr.update()
    new_idx = selected_idx + direction
    new_idx = max(0, min(new_idx, len(segments) - 1))
    return gr.update(value=f"Segmento {new_idx + 1}")


# --- Traduccion ---

def parsear_srt(contenido_srt):
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


def traducir_lote(textos_numerados, source_lang, source_code, target_lang, target_code, instrucciones_extra=""):
    extra = ""
    if instrucciones_extra.strip():
        extra = instrucciones_extra.strip()

    bloque_texto = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(textos_numerados))

    prompt = PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        source_code=source_code,
        target_lang=target_lang,
        target_code=target_code,
        instrucciones_extra=extra,
        text=bloque_texto,
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }, timeout=300)
    resp.raise_for_status()
    respuesta = resp.json()["response"].strip()

    traducciones = {}
    for match in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[|\Z)', respuesta, re.DOTALL):
        num = int(match.group(1))
        texto = match.group(2).strip()
        traducciones[num] = texto

    resultado = []
    for i, texto_original in enumerate(textos_numerados):
        resultado.append(traducciones.get(i + 1, f"[SIN TRADUCCION] {texto_original}"))
    return resultado


def traducir_srt(archivo_srt, idioma_origen, idioma_destino, instrucciones, progress=gr.Progress()):
    if not archivo_srt:
        return None, "No se subio ningun archivo SRT."

    with open(archivo_srt, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = parsear_srt(contenido)
    if not bloques:
        return None, "El archivo SRT esta vacio o tiene formato invalido."

    source_lang = idioma_origen
    source_code = IDIOMAS.get(idioma_origen, "en")
    target_lang = idioma_destino
    target_code = IDIOMAS.get(idioma_destino, "es")

    bloques_traducidos = []
    total = len(bloques)
    total_lotes = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for lote_idx in range(total_lotes):
        inicio = lote_idx * BATCH_SIZE
        fin = min(inicio + BATCH_SIZE, total)
        lote = bloques[inicio:fin]

        progress((lote_idx + 1) / total_lotes, desc=f"Traduciendo lote {lote_idx + 1}/{total_lotes} (bloques {inicio + 1}-{fin}/{total})")

        textos = [texto for _, _, texto in lote]
        try:
            traducciones = traducir_lote(textos, source_lang, source_code, target_lang, target_code, instrucciones)
        except Exception as e:
            traducciones = [f"[ERROR: {e}]"] * len(textos)

        for i, (numero, tiempo, _) in enumerate(lote):
            bloques_traducidos.append((numero, tiempo, traducciones[i]))

    nombre_base_srt = os.path.splitext(os.path.basename(archivo_srt))[0]
    output_dir = tempfile.mkdtemp()
    path_salida = os.path.join(output_dir, f"{nombre_base_srt}_{target_code}.srt")

    with open(path_salida, "w", encoding="utf-8") as f:
        for numero, tiempo, texto in bloques_traducidos:
            f.write(f"{numero}\n{tiempo}\n{texto}\n\n")

    return path_salida, f"Traduccion completada: {total} bloques en {total_lotes} lotes ({source_lang} -> {target_lang})"


# JS para play de segmento: reproduce el video desde inicio hasta fin del segmento
PLAY_SEGMENT_JS = """
(inicio, fin) => {
    const video = document.querySelector('#video_player video');
    if (!video) return;
    video.currentTime = inicio;
    video.play();
    const onTime = () => {
        if (video.currentTime >= fin) {
            video.pause();
            video.removeEventListener('timeupdate', onTime);
        }
    };
    video.removeEventListener('timeupdate', window._stopSegHandler || (() => {}));
    window._stopSegHandler = onTime;
    video.addEventListener('timeupdate', onTime);
}
"""

# ===================== INTERFAZ =====================

with gr.Blocks(title="Subtitulos v1.5") as demo:

    # --- State ---
    segments_state = gr.State([])
    selected_idx_state = gr.State(-1)
    video_path_state = gr.State("")

    gr.Markdown("# Subtitulos v1.5")

    with gr.Tab("Transcribir"):

        # ============ FILA 1: Video + Config ============
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("**Audio / Video**")
                video = gr.Video(label="Subi tu video o audio", elem_id="video_player")

            with gr.Column(scale=1):
                gr.Markdown("**Propiedades segmento**")
                with gr.Row():
                    cfg_max_dur = gr.Number(value=15.0, label="Tiempo max:", minimum=1, maximum=60, step=0.5)
                    cfg_max_pal = gr.Number(value=20, label="Palabras max:", minimum=5, maximum=50, step=1)
                cfg_cortar_antes = gr.Textbox(value=DEFAULTS_CORTAR_ANTES, label="Cortar antes de:", lines=2)
                cfg_no_dejar = gr.Textbox(value=DEFAULTS_NO_DEJAR_AL_FINAL, label="No dejar al final:", lines=1)
                cfg_puntuacion = gr.Textbox(value=DEFAULTS_PUNTUACION, label="Puntuacion de corte (separadas por espacio):", lines=1)
                with gr.Accordion("Avanzado", open=False):
                    cfg_min_pal = gr.Number(value=4, label="Min palabras (corte oracion)", minimum=1, maximum=15, step=1)
                    cfg_min_pal_suave = gr.Number(value=8, label="Min palabras (corte suave)", minimum=2, maximum=20, step=1)
                    cfg_pausa_min = gr.Number(value=0.25, label="Pausa minima (seg)", minimum=0.05, maximum=2.0, step=0.05)
                    cfg_tolerancia = gr.Number(value=5, label="Tolerancia adelante", minimum=1, maximum=15, step=1)
                btn_transcribir = gr.Button("Transcripcion", variant="primary", size="lg")

        # ============ FILA 2: Segmento + Tiempo + Texto ============
        with gr.Row(equal_height=True):
            # --- Segmento (selector + nav + play) : mismo ancho que video ---
            with gr.Column(scale=1, min_width=160):
                gr.Markdown("**Segmento**")
                seg_selector = gr.Dropdown(label="Seleccionar", choices=[], interactive=True)
                with gr.Row():
                    btn_prev = gr.Button("< Anterior", size="sm")
                    btn_next = gr.Button("Siguiente >", size="sm")
                btn_play = gr.Button("Play segmento", variant="secondary")
                seg_inicio = gr.Number(label="Inicio:", precision=3, step=0.01)
                seg_fin = gr.Number(label="Fin:", precision=3, step=0.01)
                seg_duracion = gr.Number(label="Duracion:", precision=2, interactive=False)

            # --- Texto : mismo ancho que config ---
            with gr.Column(scale=1, min_width=300):
                seg_label = gr.Markdown("**Texto**")
                seg_texto = gr.Textbox(label="Texto del segmento:", lines=8)

        # ============ FILA 3: Botones ============
        with gr.Row():
            btn_guardar = gr.Button("Guardar y Descargar", variant="secondary", size="lg")
            btn_siguiente = gr.Button("Siguiente (Traduccion)", variant="primary", size="lg")

        output_srt = gr.File(visible=False)

        # ============ EVENTS ============

        btn_transcribir.click(
            fn=transcribir,
            inputs=[video, cfg_max_dur, cfg_max_pal, cfg_min_pal, cfg_min_pal_suave,
                    cfg_pausa_min, cfg_tolerancia, cfg_cortar_antes, cfg_no_dejar, cfg_puntuacion],
            outputs=[segments_state, selected_idx_state, video_path_state,
                     seg_selector,
                     seg_inicio, seg_fin, seg_duracion,
                     seg_texto, seg_label],
        )

        seg_selector.change(
            fn=on_select_segment,
            inputs=[seg_selector, segments_state],
            outputs=[selected_idx_state, seg_inicio, seg_fin, seg_duracion,
                     seg_texto, seg_label],
        )

        seg_texto.input(
            fn=on_update_texto,
            inputs=[seg_texto, segments_state, selected_idx_state],
            outputs=[segments_state],
        )

        seg_inicio.input(
            fn=on_update_inicio,
            inputs=[seg_inicio, segments_state, selected_idx_state],
            outputs=[segments_state, seg_duracion],
        )

        seg_fin.input(
            fn=on_update_fin,
            inputs=[seg_fin, segments_state, selected_idx_state],
            outputs=[segments_state, seg_duracion],
        )

        btn_prev.click(
            fn=nav_segmento,
            inputs=[gr.Number(value=-1, visible=False), segments_state, selected_idx_state],
            outputs=[seg_selector],
        )

        btn_next.click(
            fn=nav_segmento,
            inputs=[gr.Number(value=1, visible=False), segments_state, selected_idx_state],
            outputs=[seg_selector],
        )

        btn_play.click(
            fn=None,
            inputs=[seg_inicio, seg_fin],
            js=PLAY_SEGMENT_JS,
        )

        btn_guardar.click(
            fn=guardar_srt,
            inputs=[segments_state, video_path_state],
            outputs=[output_srt],
        ).then(fn=lambda: gr.update(visible=True), outputs=[output_srt])

    # --- Tab Traducir ---
    with gr.Tab("Traducir SRT"):
        with gr.Row():
            input_srt = gr.File(label="Subi tu archivo SRT", file_types=[".srt"])

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

        instrucciones = gr.Textbox(
            label="Instrucciones extra (opcional)",
            placeholder='Ej: Translate "muffins" as "magdalenas". Use informal tone (tuteo).',
            lines=3,
        )

        btn_traducir = gr.Button("Traducir SRT", variant="primary")

        with gr.Row():
            output_traducido = gr.File(label="SRT Traducido")
            output_estado = gr.Textbox(label="Estado", interactive=False)

        btn_traducir.click(
            fn=traducir_srt,
            inputs=[input_srt, idioma_origen, idioma_destino, instrucciones],
            outputs=[output_traducido, output_estado],
        )

    # "Siguiente" guarda SRT y lo pasa a la tab de traduccion
    btn_siguiente.click(
        fn=guardar_srt,
        inputs=[segments_state, video_path_state],
        outputs=[input_srt],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3838)
