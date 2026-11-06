"""Editor de hablantes: página aparte (/editor) para corregir quién dice cada loop mirando el video.

El video se abre desde la compu de quien edita (no se sube). Se puede cambiar el hablante de
cada loop, corregir su texto, unirlo con el siguiente o borrarlo. Los hablantes van a las
palabras de <video>_<idioma>.<motor>.hablantes.json y el texto corregido a las palabras del
.json, así "Rearmar" respeta las dos cosas (las uniones no: Rearmar arma los loops de cero).
Se reescriben el SRT limpio y el .nombres.srt con los mismos loops.
"""
import json
import os
import re
from collections import Counter, defaultdict
from types import SimpleNamespace

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

import hablantes
import loops
import server_translate as st

router = APIRouter()
PAGINA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "editor.html")


def _con_hablantes():
    """Transcripciones que tienen hablantes identificados, la más nueva primero."""
    return [n for n in st.transcripciones() if os.path.exists(st.hablantes_json(n))]


def _validar(srt):
    if srt not in _con_hablantes():
        raise HTTPException(404, "Esa transcripción no existe o no tiene hablantes identificados.")


def _segundos(texto):
    h, m, s = texto.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _cargar(srt):
    """(resultado, palabras, etiquetas, datos, grupos).

    resultado es el JSON de stable-whisper tal cual y palabras sus dicts {"word", "start", "end"...}
    en orden; grupos son los (desde, hasta) de palabras de cada loop del SRT.
    """
    with open(st.palabras_json(srt), encoding="utf-8") as f:
        resultado = json.load(f)
    palabras = [w for s in resultado["segments"] for w in s["words"]]
    datos = hablantes.cargar(st.hablantes_json(srt))
    if len(datos["palabras"]) != len(palabras):
        raise HTTPException(409, "Las palabras y los hablantes no coinciden: volvé a transcribir el video.")
    with open(os.path.join(st.OUTPUT_DIR, srt), encoding="utf-8") as f:
        bloques = [b.splitlines() for b in re.split(r"\n\s*\n", f.read().strip())]
    grupos, i = [], 0
    for b in bloques:
        if len(b) < 3:
            continue
        # El texto del loop es la unión de sus palabras: se avanza hasta cubrir sus letras
        objetivo = re.sub(r"\s", "", " ".join(b[2:]))
        if i < len(palabras) and not objetivo.startswith(re.sub(r"\s", "", palabras[i]["word"])):
            objetivo = re.sub(r"\s", "", re.sub(r"^[^:]{1,40}:\s*", "", " ".join(b[2:])))  # SRT viejo con nombres
        desde, letras = i, 0
        while i < len(palabras) and letras < len(objetivo):
            letras += len(re.sub(r"\s", "", palabras[i]["word"]))
            i += 1
        grupos.append((desde, i))
    if i != len(palabras):
        raise HTTPException(409, "El SRT no coincide con las palabras guardadas: tocá \"Rearmar\" y volvé a abrir el editor.")
    return resultado, palabras, datos["palabras"], datos, grupos


def _texto(palabras):
    return " ".join("".join(w["word"] for w in palabras).split())


def _corregir(viejas, texto):
    """Palabras nuevas para el texto corregido de un loop.

    Con la misma cantidad de palabras cada una conserva sus tiempos; si no, se reparten
    entre el inicio y el fin del loop según el largo de cada palabra.
    """
    nuevas = texto.split()
    if len(nuevas) == len(viejas):
        return [{**v, "word": " " + n} for v, n in zip(viejas, nuevas)]
    inicio, fin = viejas[0]["start"], viejas[-1]["end"]
    total, hecho, salida = sum(len(n) for n in nuevas), 0, []
    for n in nuevas:
        a = inicio + (fin - inicio) * hecho / total
        hecho += len(n)
        salida.append({"word": " " + n, "start": round(a, 3), "end": round(inicio + (fin - inicio) * hecho / total, 3),
                       "probability": 1.0, "tokens": []})
    return salida


@router.get("/editor")
def pagina():
    return FileResponse(PAGINA, headers={"Cache-Control": "no-cache"})


@router.get("/api/editor/transcripciones")
def transcripciones():
    return _con_hablantes()


@router.get("/api/editor/datos")
def datos(srt: str):
    _validar(srt)
    _, palabras, etiquetas, info, grupos = _cargar(srt)
    lista = [{"inicio": palabras[a]["start"], "fin": palabras[b - 1]["end"], "texto": _texto(palabras[a:b]),
              "hablante": Counter(etiquetas[a:b]).most_common(1)[0][0]} for a, b in grupos]
    return {"loops": lista, "todos": info.get("todos", True),
            "hablantes": {h: {"nombre": d["nombre"], "segundos": d["segundos"], "muestra": bool(d.get("muestra"))}
                          for h, d in info["hablantes"].items()}}


@router.get("/api/editor/muestra")
def muestra(srt: str, h: str):
    _validar(srt)
    ruta = (hablantes.cargar(st.hablantes_json(srt))["hablantes"].get(h) or {}).get("muestra")
    if not ruta or not os.path.exists(ruta):
        raise HTTPException(404, "Ese hablante no tiene muestra.")
    return FileResponse(ruta, media_type="audio/wav")


@router.get("/api/editor/srt")
def descargar(srt: str):
    _validar(srt)
    return FileResponse(st.srt_nombres(srt), filename=os.path.basename(st.srt_nombres(srt)))


@router.post("/api/editor/guardar")
def guardar(srt: str, cambios: dict = Body(...)):
    """cambios: {"loops": [{"ids": [3, 4], "hablante": "H1", "texto": "..."}], "nombres": {"H1": "Daria"}, "todos": bool}

    ids son los loops originales que forman cada uno (más de uno si se unieron); texto vacío = borrar el loop.
    """
    _validar(srt)
    resultado, palabras, _, info, grupos = _cargar(srt)
    pedidos, nombres = cambios.get("loops", []), cambios.get("nombres", {})
    ids = [i for l in pedidos for i in l.get("ids", [])]
    if (ids != list(range(len(grupos))) or not all(re.fullmatch(r"H\d+", l.get("hablante", "")) for l in pedidos)
            or not all(l["ids"] for l in pedidos)):
        raise HTTPException(400, "Los loops cambiaron mientras editabas: recargá la página.")

    # Palabras, hablantes y loops nuevos, en orden
    nuevas, etiquetas, con_hablante, segmentos, texto_cambiado = [], [], [], [], False
    for l in pedidos:
        ws = palabras[grupos[l["ids"][0]][0]:grupos[l["ids"][-1]][1]]
        texto = " ".join(str(l.get("texto", "")).split())
        if not texto:  # borrado
            texto_cambiado = True
            continue
        if texto != _texto(ws):
            ws, texto_cambiado = _corregir(ws, texto), True
        nuevas += ws
        etiquetas += [l["hablante"]] * len(ws)
        con_hablante.append((l["hablante"], [SimpleNamespace(word=w["word"], start=w["start"], end=w["end"]) for w in ws]))
        segmentos.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": "".join(w["word"] for w in ws),
                          "seek": 0, "tokens": [], "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 0.0,
                          "no_speech_prob": 0.0, "words": ws})
    if not con_hablante:
        raise HTTPException(400, "No se pueden borrar todos los loops.")

    if texto_cambiado:
        # Un segmento por loop: así las palabras corregidas quedan en el .json y Rearmar las usa
        resultado["segments"] = segmentos
        resultado["text"] = "".join(s["text"] for s in segmentos)
        with open(st.palabras_json(srt), "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False)

    # Hablantes que quedaron (los nuevos sin muestra; los que se quedaron sin palabras se van)
    segundos = defaultdict(float)
    for w, h in zip(nuevas, etiquetas):
        segundos[h] += w["end"] - w["start"]
    primera = {h: "".join(w.word for w in ws).strip() for h, ws in reversed(con_hablante)}
    quedan = {}
    for h in sorted(segundos, key=lambda x: int(x[1:])):
        d = info["hablantes"].get(h) or {"nombre": "", "frase": primera.get(h, ""), "muestra": None}
        d["segundos"] = round(segundos[h], 1)
        if h in nombres:
            d["nombre"] = (nombres[h] or "").strip()
        quedan[h] = d
    info.update(palabras=etiquetas, hablantes=quedan, todos=bool(cambios.get("todos", info.get("todos", True))))
    with open(st.hablantes_json(srt), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    lista = [ws for _, ws in con_hablante]
    loops.escribir_srt(lista, st.srt_nombres(srt),
                       loops.nombres_por_loop(con_hablante, {h: d["nombre"] for h, d in quedan.items()}, info["todos"]))
    loops.escribir_srt(lista, os.path.join(st.OUTPUT_DIR, srt))  # el limpio (los SRT viejos tenían los nombres acá)
    return {"ok": True, "archivo": os.path.basename(st.srt_nombres(srt))}
