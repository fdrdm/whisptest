import stable_whisper

# Esto descargará el modelo 'base' la primera vez (aprox 150MB)
model = stable_whisper.load_model('base')

# Transcribir con timestamps estabilizados
# (Cambia 'audio.mp3' por un archivo que tengas)
result = model.transcribe('audio.mp3')

# Generar un SRT con las palabras resaltadas (estilo karaoke)
result.to_srt_vtt('subtitulos.srt', word_level=True)

print("¡Subtítulos generados con éxito!")
