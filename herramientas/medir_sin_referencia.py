"""Mide las tres transcripciones de un video entre sí, sin subtítulos de referencia.

Uso:
  .venv/bin/python herramientas/medir_sin_referencia.py "output_translate/palabras/<video>_<idioma>"

Lee <base>.whisper / .nemotron / .deepinfra (.json de palabras + .hablantes.json) y saca, por motor:
- Texto: % de palabras que coinciden con cada uno de los otros dos y cuántos tramos tiene en los que
  es el único que difiere (los otros dos coinciden entre sí). Un error repetido (un nombre) cuenta cada vez.
- Sincronía: entre palabras iguales de dos motores, cuántas empiezan a menos de 0,3 s una de otra.
- Turnos: % de cambios de hablante que caen a mitad de frase (la palabra anterior no termina con
  . ? ! y la pausa es menor a 0,3 s). Cuanto más bajo, mejor corta.
- Identidad: con la voz de pyannote (la corrida de Whisper) como referencia, % de palabras de cada
  hablante que son de su voz mayoritaria (un hablante = una voz) y % de palabras de cada voz que
  cayeron en su hablante mayoritario (una voz = un hablante). Para Whisper es 100% por definición.
- Loops: cantidad y cuántos duran menos de 1 s (del SRT limpio).
"""
import bisect
import collections
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparar_motores import normalizar  # noqa: E402

MOTORES = ("whisper", "nemotron", "deepinfra")


def cargar(base, m):
    j = json.load(open(f"{base}.{m}.json", encoding="utf-8"))
    ws = [w for s in j["segments"] for w in s["words"]]
    et = json.load(open(f"{base}.{m}.hablantes.json", encoding="utf-8"))["palabras"]
    return [{"word": w["word"].strip(), "start": w["start"], "end": w["end"], "h": e} for w, e in zip(ws, et)]


def tokens(ws):
    """[(token normalizado, índice de palabra)]"""
    return [(t, i) for i, w in enumerate(ws) for t in normalizar(w["word"])]


def emparejar(a, b):
    """Pares (i, j) de palabras iguales y en orden entre dos motores."""
    ta, tb = tokens(a), tokens(b)
    pares = []
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, [x for x, _ in ta], [x for x, _ in tb], autojunk=False).get_opcodes():
        if op == "equal":
            pares += [(ta[a1 + k][1], tb[b1 + k][1]) for k in range(a2 - a1)]
    return pares, len(ta), len(tb)


def main():
    base = sys.argv[1]
    W = {m: cargar(base, m) for m in MOTORES if os.path.exists(f"{base}.{m}.json")}
    nombres = list(W)
    pares = {}
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            pares[(a, b)] = emparejar(W[a], W[b])

    print(f"{'':10} {'palabras':>8} {'acuerdo con los otros':>22} {'solo él difiere':>16} {'sincronía ±0,3 s':>17} {'cortes a mitad de frase':>24} {'hablantes':>10} {'1 habl.=1 voz':>14} {'1 voz=1 habl.':>14} {'loops':>6} {'<1 s':>5}")
    voz = None
    if "whisper" in W:
        tw = [(w["start"] + w["end"]) / 2 for w in W["whisper"]]

        def voz(t):
            i = bisect.bisect_left(tw, t)
            c = [k for k in (i - 1, i) if 0 <= k < len(tw)]
            k = min(c, key=lambda k: abs(tw[k] - t))
            return W["whisper"][k]["h"] if abs(tw[k] - t) < 1 else None

    for m in nombres:
        ws = W[m]
        otros = [o for o in nombres if o != m]
        # Acuerdo y sincronía con cada otro motor
        acuerdos, sinc = [], []
        for o in otros:
            (p, na, nb) = pares[(m, o)] if (m, o) in pares else pares[(o, m)]
            if (m, o) not in pares:
                p = [(j, i) for i, j in p]
            acuerdos.append(2 * len(p) / (na + nb))
            sinc.append(sum(abs(ws[i]["start"] - W[o][j]["start"]) < 0.3 for i, j in p) / len(p))
        # Solo él difiere: palabras de m sin pareja en ninguno de los otros dos, en tramos seguidos
        con_pareja = {o: {i for i, _ in (pares[(m, o)][0] if (m, o) in pares else [(j, i) for i, j in pares[(o, m)][0]])} for o in otros}
        solo, k = 0, 0
        while k < len(ws):
            if all(k not in con_pareja[o] for o in otros):
                solo += 1
                while k < len(ws) and all(k not in con_pareja[o] for o in otros):
                    k += 1
            else:
                k += 1
        # Cortes a mitad de frase
        cambios = [i for i in range(1, len(ws)) if ws[i]["h"] != ws[i - 1]["h"]]
        mitad = sum(1 for i in cambios if not re.search(r"[.?!…]$", ws[i - 1]["word"]) and ws[i]["start"] - ws[i - 1]["end"] < 0.3)
        # Identidad por voz de pyannote
        pureza = entera = float("nan")
        if voz:
            por, inv = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
            for w in ws:
                v = voz((w["start"] + w["end"]) / 2)
                if v:
                    por[w["h"]][v] += 1
                    inv[v][w["h"]] += 1
            tot = sum(sum(c.values()) for c in por.values())
            pureza = sum(c.most_common(1)[0][1] for c in por.values()) / tot
            entera = sum(c.most_common(1)[0][1] for c in inv.values()) / tot
        # Loops del SRT limpio
        srt = re.sub(r"/palabras/", "/", base) + f".{m}.srt"
        dur = []
        if os.path.exists(srt):
            def seg(t):
                h, mi, s = t.replace(",", ".").split(":")
                return int(h) * 3600 + int(mi) * 60 + float(s)
            dur = [seg(z) - seg(a) for a, z in re.findall(r"(\S+) --> (\S+)", open(srt, encoding="utf-8").read())]
        print(f"{m:10} {len(ws):8} {' / '.join(f'{x:.0%}' for x in acuerdos):>22} {solo:16} {' / '.join(f'{x:.0%}' for x in sinc):>17} "
              f"{mitad / max(len(cambios), 1):23.0%} {len(set(w['h'] for w in ws)):10} {pureza:14.0%} {entera:14.0%} {len(dur):6} {sum(d < 1 for d in dur):5}")


if __name__ == "__main__":
    main()
