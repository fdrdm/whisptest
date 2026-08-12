import gradio as gr
import stable_whisper

# Usamos large para los signos de puntuación perfectos
model = stable_whisper.load_model('turbo')

def segundos_a_srt(segundos):
    """Convierte los segundos al formato de tiempo que usan los archivos SRT"""
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos - int(segundos)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def procesar(input_file):
    if not input_file:
        return None

    # 1. Transcripción cruda
    result = model.transcribe(input_file)
    idioma_detectado = result.language or "autodetectado"
    path_srt = f"subtitulos_{idioma_detectado}.srt"
    
    # 2. Extraemos TODAS las palabras
    todas_las_palabras = []
    for segment in result.segments:
        for word in segment.words:
            todas_las_palabras.append(word)
            
    # 3. Armador de Loops (Relajado y ajustado)
    loops = []
    loop_actual = []
    
    for i, word in enumerate(todas_las_palabras):
        loop_actual.append(word)
        
        cantidad_palabras = len(loop_actual)
        tiempo_inicio = loop_actual[0].start
        tiempo_fin = loop_actual[-1].end
        duracion = tiempo_fin - tiempo_inicio
        
        texto_limpio = word.word.strip()
        termina_en_puntuacion = texto_limpio.endswith(('.', '?', '!', ',', ':'))
        
        # --- NUEVAS CONDICIONES DE CORTE ---
        # A. Límite estricto: 25 palabras o 20 segundos máximo
        limite_alcanzado = (cantidad_palabras >= 25 or duracion >= 20.0)
        
        # B. Corte lógico: Hay una pausa (puntuación) y el loop ya tiene al menos 4 palabras
        # (Mantenemos el mínimo de 4 acá para que no corte si dice "Hola, ", pero permitimos
        # que queden loops cortos al final si se da el caso).
        corte_logico = termina_en_puntuacion and cantidad_palabras >= 4
        
        es_ultima_palabra = (i == len(todas_las_palabras) - 1)
        
        # Si llegamos a la última palabra del video, guardamos lo que haya sin importar el largo
        if es_ultima_palabra:
            loops.append(loop_actual)
            break
            
        # Si se cumple el límite o hay un buen corte por puntuación, cerramos el loop
        if limite_alcanzado or corte_logico:
            loops.append(loop_actual)
            loop_actual = [] # Reseteamos para el próximo

    # 4. Escribir el archivo SRT
    with open(path_srt, "w", encoding="utf-8") as f:
        for i, loop in enumerate(loops):
            inicio_srt = segundos_a_srt(loop[0].start)
            fin_srt = segundos_a_srt(loop[-1].end)
            
            texto_loop = "".join([w.word for w in loop]).strip()
            
            f.write(f"{i + 1}\n")
            f.write(f"{inicio_srt} --> {fin_srt}\n")
            f.write(f"{texto_loop}\n\n")

    return path_srt

# Interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Generador de Loops (Máx 25 pal / 20 seg) 🎬")
    
    with gr.Row():
        input_data = gr.Video(label="Subí tu Video o Audio")
    
    btn = gr.Button("Generar SRT de Loops", variant="primary")
    
    with gr.Row():
        output_srt = gr.File(label="Archivo SRT")

    btn.click(
        fn=procesar, 
        inputs=[input_data], 
        outputs=[output_srt]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8188)
