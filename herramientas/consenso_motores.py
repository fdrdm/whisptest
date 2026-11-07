"""Compara transcripciones entre sí, sin referencia: cuando dos motores coinciden y el tercero no, el
tercero casi seguro se equivocó. Sirve cuando no hay subtítulos confiables (YouTube automático).

Uso:
  .venv/bin/python herramientas/consenso_motores.py nombre=archivo.json nombre=archivo.json [nombre=archivo.json ...]

Cada archivo es un JSON de palabras de stable-ts (palabras/<video>.<motor>.json) o una respuesta cruda
de DeepInfra ("words"). Normaliza igual que comparar_motores.py.

Con tres o más motores imprime, por motor, cuántas veces fue el único en discrepar ("errores propios"),
y una muestra de esas discrepancias con el minuto y lo que escribieron los demás.
"""
import difflib
import itertools
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from comparar_motores import normalizar  # noqa: E402


def palabras(path):
    d = json.load(open(path, encoding="utf-8"))
    ws = d["words"] if "words" in d else [w for s in d["segments"] for w in s["words"]]
    salida = []
    for w in ws:
        for t in normalizar(w["word"]):
            salida.append((t, w["start"]))
    return salida


def alinear(a, b):
    """Para cada posición de a, la posición equivalente en b (o None si a tiene algo que b no)."""
    mapa = [None] * len(a)
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, [x for x, _ in a], [x for x, _ in b], autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(a2 - a1):
                mapa[a1 + k] = b1 + k
    return mapa


def main():
    motores = dict(arg.split("=", 1) for arg in sys.argv[1:])
    H = {m: palabras(p) for m, p in motores.items()}
    nombres = list(H)
    # Acuerdo de a pares: % de palabras del primero que aparecen igual y en orden en el segundo
    print("Acuerdo de a pares (palabras iguales y en el mismo orden):")
    for a, b in itertools.combinations(nombres, 2):
        iguales = sum(x is not None for x in alinear(H[a], H[b]))
        print(f"  {a} ~ {b}: {iguales} de {len(H[a])} / {len(H[b])}  ({2 * iguales / (len(H[a]) + len(H[b])):.1%})")
    if len(nombres) < 3:
        return

    # Errores propios: tramos donde un motor difiere de los otros dos y los otros dos coinciden entre sí
    print("\nErrores propios (los otros dos coinciden y este no), aproximado:")
    for m in nombres:
        otros = [o for o in nombres if o != m]
        mapa_mo = {o: alinear(H[m], H[o]) for o in otros}
        mapa_oo = alinear(H[otros[0]], H[otros[1]])
        propios, muestra = 0, []
        i = 0
        while i < len(H[m]):
            # tramo seguido de palabras de m que no están en ninguno de los otros dos
            if all(mapa_mo[o][i] is None for o in otros):
                j = i
                while j < len(H[m]) and all(mapa_mo[o][j] is None for o in otros):
                    j += 1
                # ¿los otros dos coinciden entre sí alrededor? (la palabra anterior de m en ambos, y esos puntos se corresponden)
                ant = next((k for k in range(i - 1, -1, -1) if all(mapa_mo[o][k] is not None for o in otros)), None)
                if ant is not None and mapa_oo[mapa_mo[otros[0]][ant]] == mapa_mo[otros[1]][ant]:
                    propios += 1
                    if len(muestra) < 12:
                        p0 = mapa_mo[otros[0]][ant]
                        texto_otro = " ".join(x for x, _ in H[otros[0]][p0 + 1:p0 + 1 + max(1, j - i)])
                        muestra.append((H[m][i][1], " ".join(x for x, _ in H[m][i:j]), texto_otro))
                i = j
            else:
                i += 1
        print(f"== {m}: {propios} tramos propios")
        for t, mio, otro in muestra:
            print(f"   {t / 60:5.2f} min  escribió {mio!r:40} · los otros: {otro!r}")


if __name__ == "__main__":
    main()
