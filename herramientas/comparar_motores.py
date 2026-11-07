"""Compara las transcripciones de un video contra subtítulos de referencia, con una medición justa.

Uso:
  .venv/bin/python herramientas/comparar_motores.py REFERENCIA.srt "output_translate/palabras/<video>_<idioma>" [motor=archivo.json ...]

Sin motores explícitos busca <base>.whisper.json, <base>.nemotron.json y <base>.deepinfra.crudo.json.
Un motor explícito puede ser un JSON de palabras de stable-ts o una respuesta cruda de DeepInfra ("words").

Qué hace justa a la medición (sin esto, los subtítulos castigan a los motores que transcriben literal):
- saca las muletillas de los dos lados (uh, um, hm...), porque los subtítulos no las escriben;
- unifica gonna/going to, wanna/want to, números en letras y cifras, "|" del OCR como "I";
- no cuenta como error una diferencia de espacios ("light bulb" / "lightbulb", OCR que pegó palabras).

Imprime, por motor: WER (sustituciones + borradas + de más, sobre las palabras de la referencia),
las frases de 3+ palabras perdidas o agregadas, y los errores que solo tiene ese motor (los otros
dos escribieron lo que dice la referencia; aproximado).
"""
import collections
import difflib
import json
import re
import sys

MULETILLAS = {"uh", "um", "hm", "hmm", "mm", "mhm", "ah", "oh", "eh", "er", "huh", "ooh", "aah", "ee", "oof", "ugh",
              "whoa", "woo", "hey"}
NUMEROS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
           "nine": "9", "ten": "10", "twelve": "12", "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100",
           "thousand": "1000"}


def normalizar(texto):
    t = texto.lower().replace("’", "'").replace("''", "'").replace("|", "i").replace("-", " ")
    for a, b in (("gonna", "going to"), ("wanna", "want to"), ("gotta", "got to"), ("'cause", "because"),
                 ("kinda", "kind of"), ("ya", "you")):
        t = re.sub(rf"\b{a}\b", b, t)
    return [NUMEROS.get(w, w) for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t) if w not in MULETILLAS]


def referencia(path):
    palabras = []
    for bloque in re.split(r"\n\s*\n", open(path, encoding="utf-8").read().strip()):
        lineas = bloque.splitlines()
        if len(lineas) >= 3 and "-->" in lineas[1]:
            palabras += normalizar(re.sub(r"\[[^\]]*\]|\([^)]*\)", "", " ".join(lineas[2:])))  # sin [sonidos]
    return palabras


def hipotesis(path):
    d = json.load(open(path, encoding="utf-8"))
    palabras = d["words"] if "words" in d else [w for s in d["segments"] for w in s["words"]]
    return normalizar(" ".join(w["word"] for w in palabras))


def pegar(ws):
    return "".join(ws).replace("'", "")


def comparar(ref, hip):
    """(S, D, I, errores) con tolerancia a espacios; errores = [(op, texto ref, texto hip)]."""
    S = D = I = 0
    errores = []
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, ref, hip, autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and pegar(ref[a1:a2]) == pegar(hip[b1:b2])):
            continue
        if op == "replace":
            k = min(a2 - a1, b2 - b1)
            S += k
            D += (a2 - a1) - k
            I += (b2 - b1) - k
        elif op == "delete":
            D += a2 - a1
        else:
            I += b2 - b1
        errores.append((op, " ".join(ref[a1:a2]), " ".join(hip[b1:b2])))
    return S, D, I, errores


def main():
    ref = referencia(sys.argv[1])
    base = sys.argv[2]
    motores = dict(a.split("=", 1) for a in sys.argv[3:]) or {
        "whisper": base + ".whisper.json", "nemotron": base + ".nemotron.json", "deepinfra": base + ".deepinfra.crudo.json"}
    H = {m: hipotesis(p) for m, p in motores.items()}
    resultados = {m: comparar(ref, h) for m, h in H.items()}

    print(f"Referencia: {len(ref)} palabras normalizadas\n")
    print(f"{'motor':12} {'palabras':>8} {'WER':>7} {'sust':>5} {'borradas':>9} {'de más':>7}")
    for m, (S, D, I, _) in resultados.items():
        print(f"{m:12} {len(H[m]):8} {(S + D + I) / len(ref):7.1%} {S:5} {D:9} {I:7}")

    print("\nFrases perdidas o agregadas (3+ palabras):")
    for m, (_, _, _, errores) in resultados.items():
        print(f"== {m}")
        for op, r, h in errores:
            if max(len(r.split()), len(h.split())) >= 3:
                print(f"   {op:7} ref: {r[:70]!r:75} -> {h[:70]!r}")

    if len(H) >= 3:
        print("\nErrores que solo tiene ese motor (aproximado):")
        texto = {m: " " + " ".join(h) + " " for m, h in H.items()}
        for m, (_, _, _, errores) in resultados.items():
            solos = [(r, h) for op, r, h in errores if r and all((" " + r + " ") in texto[o] for o in H if o != m)]
            print(f"== {m}: {len(solos)}. Ej.: " + " | ".join(f"{r!r}->{h!r}" for r, h in solos[:10]))


if __name__ == "__main__":
    main()
