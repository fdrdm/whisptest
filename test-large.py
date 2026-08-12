import stable_whisper

# Cargamos el modelo
model = stable_whisper.load_model('base')

# Transcribimos (acordate de que acá podés forzar el idioma original con language='en')
result = model.transcribe('audio.mp3')

# Generamos el archivo SRT
nombre_archivo = 'subtitulos.srt'
result.to_srt_vtt(nombre_archivo, word_level=False)

print("¡Subtítulos generados con éxito!\n")
print(f"--- Mostrando contenido de {nombre_archivo} ---")

# Esto hace exactamente lo mismo que 'cat' en la terminal
with open(nombre_archivo, 'r', encoding='utf-8') as f:
    print(f.read())
