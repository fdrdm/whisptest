
import collections
import datetime

# Rango de fechas: desde el 1 de junio de 2026 hasta hoy
start_date = datetime.date(2026, 6, 1)
today = datetime.date(2026, 6, 9)

counts = collections.defaultdict(lambda: {"SEPARACION": 0, "TRADUCCION": 0})

try:
    with open("uso.log", "r") as f:
        for line in f:
            if line.startswith("2026-06"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    datetime_part = parts[0]
                    try:
                        date_str = datetime_part.split(" ")[0]
                        action = parts[1]
                        if action in ["SEPARACION", "TRADUCCION"]:
                            counts[date_str][action] += 1
                    except Exception:
                        continue
except FileNotFoundError:
    print("uso.log no encontrado")
    exit(1)

print("| Fecha | Separaciones | Traducciones |")
print("|-------|--------------|--------------|")

current_date = start_date
while current_date <= today:
    d_str = current_date.strftime("%Y-%m-%d")
    c = counts.get(d_str, {"SEPARACION": 0, "TRADUCCION": 0})
    print(f"| {d_str} | {c['SEPARACION']} | {c['TRADUCCION']} |")
    current_date += datetime.timedelta(days=1)
