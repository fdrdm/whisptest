"""Editor de hablantes: página aparte (/editor) para corregir quién dice cada loop mirando el video.

El video se abre desde la compu de quien edita (no se sube). Las correcciones van a las
palabras de <video>_<idioma>.hablantes.json, así "Rearmar" las respeta, y se reescribe el
.nombres.srt con los mismos loops que el SRT limpio.
"""
import json
import os
import re
from collections import Counter, defaultdict

import stable_whisper
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
    """(palabras, etiquetas, datos, grupos): grupos son los (desde, hasta) de palabras de cada loop del SRT."""
    resultado = stable_whisper.WhisperResult(st.palabras_json(srt))
    palabras = [w for s in resultado.segments for w in s.words]
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
        if i < len(palabras) and not objetivo.startswith(re.sub(r"\s", "", palabras[i].word)):
            objetivo = re.sub(r"\s", "", re.sub(r"^[^:]{1,40}:\s*", "", " ".join(b[2:])))  # SRT viejo con nombres
        desde, letras = i, 0
        while i < len(palabras) and letras < len(objetivo):
            letras += len(re.sub(r"\s", "", palabras[i].word))
            i += 1
        grupos.append((desde, i))
    if i != len(palabras):
        raise HTTPException(409, "El SRT no coincide con las palabras guardadas: tocá \"Rearmar\" y volvé a abrir el editor.")
    return palabras, datos["palabras"], datos, grupos


@router.get("/editor")
def pagina():
    return FileResponse(PAGINA, headers={"Cache-Control": "no-cache"})


@router.get("/api/editor/transcripciones")
def transcripciones():
    return _con_hablantes()


@router.get("/api/editor/datos")
def datos(srt: str):
    _validar(srt)
    palabras, etiquetas, info, grupos = _cargar(srt)
    lista = [{"inicio": palabras[a].start, "fin": palabras[b - 1].end,
              "texto": "".join(w.word for w in palabras[a:b]).strip(),
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
    """cambios: {"por_loop": ["H1", ...], "nombres": {"H1": "Daria"}, "todos": bool}"""
    _validar(srt)
    palabras, etiquetas, info, grupos = _cargar(srt)
    por_loop, nombres = cambios.get("por_loop", []), cambios.get("nombres", {})
    if len(por_loop) != len(grupos) or not all(re.fullmatch(r"H\d+", h) for h in por_loop):
        raise HTTPException(400, "Los loops cambiaron mientras editabas: recargá la página.")
    for (a, b), h in zip(grupos, por_loop):
        etiquetas[a:b] = [h] * (b - a)

    # Hablantes que quedaron (los nuevos sin muestra; los que se quedaron sin palabras se van)
    segundos = defaultdict(float)
    for w, h in zip(palabras, etiquetas):
        segundos[h] += w.end - w.start
    primera = {h: "".join(w.word for w in palabras[a:b]).strip() for (a, b), h in reversed(list(zip(grupos, por_loop)))}
    nuevos = {}
    for h in sorted(segundos, key=lambda x: int(x[1:])):
        d = info["hablantes"].get(h) or {"nombre": "", "frase": primera.get(h, ""), "muestra": None}
        d["segundos"] = round(segundos[h], 1)
        if h in nombres:
            d["nombre"] = (nombres[h] or "").strip()
        nuevos[h] = d
    info.update(palabras=etiquetas, hablantes=nuevos, todos=bool(cambios.get("todos", info.get("todos", True))))
    with open(st.hablantes_json(srt), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    con_hablante = [(h, palabras[a:b]) for (a, b), h in zip(grupos, por_loop)]
    lista = [l for _, l in con_hablante]
    loops.escribir_srt(lista, st.srt_nombres(srt),
                       loops.nombres_por_loop(con_hablante, {h: d["nombre"] for h, d in nuevos.items()}, info["todos"]))
    loops.escribir_srt(lista, os.path.join(st.OUTPUT_DIR, srt))  # el limpio (los SRT viejos tenían los nombres acá)
    return {"ok": True, "archivo": os.path.basename(st.srt_nombres(srt))}
