
import re

def get_seconds(text):
    match = re.search(r'audio=(\d+)s', text)
    if match:
        return int(match.group(1))
    return 0

file_durations = {}

# Primero pasamos para mapear nombres de archivos a duraciones desde TRANSCRIPCION y SEPARACION
with open("uso.log", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            action = parts[1]
            filename = parts[2]
            if action in ["TRANSCRIPCION", "SEPARACION"]:
                sec = get_seconds(line)
                if sec > 0:
                    # Guardamos la duración para el nombre base del archivo
                    base_name = filename.replace("_en.srt", "").replace(".srt", "")
                    file_durations[base_name] = sec

print("| Fecha | Hora | Tipo | Archivo | Minutos |")
print("|-------|------|------|----------|---------|")

with open("uso.log", "r") as f:
    for line in f:
        if line.startswith("2026-06"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                date_time = parts[0]
                date, time = date_time.split(" ")
                action = parts[1]
                filename = parts[2]
                
                if action in ["SEPARACION", "TRADUCCION"]:
                    seconds = 0
                    if action == "SEPARACION":
                        seconds = get_seconds(line)
                    else:
                        # Para traducción, buscamos en nuestro mapa de duraciones
                        base_name = filename.replace("_en.srt", "").replace(".srt", "")
                        seconds = file_durations.get(base_name, 0)
                    
                    minutes = round(seconds / 60, 2)
                    print(f"| {date} | {time} | {action} | {filename[:50]} | {minutes} |")
