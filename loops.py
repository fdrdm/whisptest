"""Armado de loops (bloques del SRT) a partir de las palabras con tiempos de Whisper.

Las palabras de cada transcripción se guardan en JSON (output_translate/palabras/),
así los loops se pueden rearmar con otros ajustes sin volver a transcribir.
"""
import re
from dataclasses import dataclass


@dataclass
class Ajustes:
    max_palabras: int = 20        # tope duro
    max_segundos: float = 15.0    # tope duro
    min_palabras: int = 4         # mínimo para cortar en fin de oración
    pausa_suave: float = 0.25     # segundos de silencio para validar corte suave
    min_palabras_suave: int = 8   # mínimo para cortar antes de conjunción/preposición
    tolerancia_adelante: int = 5  # palabras que se buscan adelante para cerrar la oración


# Conjunciones y preposiciones que marcan inicio de cláusula
CORTE_SUAVE = {
    # Conjunciones
    'and', 'or', 'but', 'so', 'yet', 'nor', 'because', 'although', 'though',
    'while', 'when', 'if', 'since', 'unless', 'until', 'whether',
    'however', 'moreover', 'nevertheless', 'otherwise', 'whereas', 'meanwhile',
    # Preposiciones que inician cláusula
    'with', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
    'between', 'against', 'without', 'beyond', 'despite', 'albeit',
}

# Conjunciones que no deben quedar colgadas al final de un bloque
CONJUNCIONES = {'and', 'or', 'but', 'so', 'yet', 'nor'}



def armar(palabras, aj=None):
    """Agrupa las palabras (con .word, .start, .end) en loops según los ajustes."""
    aj = aj or Ajustes()
    loops = []
    loop_actual = []
    total_palabras = len(palabras)
    i = 0

    while i < total_palabras:
        word = palabras[i]
        palabra = word.word.strip().lower().rstrip('.,!?:;')

        # Antes de agregar: cortar si el bloque anterior terminó en fin de oración
        # y la palabra nueva empieza con mayúscula (= oración nueva)
        if loop_actual and len(loop_actual) >= 2:
            ultima = loop_actual[-1].word.strip()
            primera_letra = word.word.strip()[:1]
            if ultima.endswith(('.', '?', '!')) and primera_letra.isupper():
                loops.append(loop_actual)
                loop_actual = []

        # Antes de agregar: si viene conjunción/preposición, el bloque tiene suficiente,
        # Y hay una pausa real en el audio (no cortar en medio de habla fluida)
        if loop_actual and len(loop_actual) >= aj.min_palabras_suave and palabra in CORTE_SUAVE:
            pausa = word.start - loop_actual[-1].end
            if pausa >= aj.pausa_suave:
                # Si la última palabra del bloque es una conjunción suelta ("but", "and"...),
                # moverla al bloque nuevo para que no quede colgada
                ultima_palabra = loop_actual[-1].word.strip().lower().rstrip('.,!?:;')
                if ultima_palabra in CONJUNCIONES and len(loop_actual) > 1:
                    colgada = loop_actual.pop()
                    loops.append(loop_actual)
                    loop_actual = [colgada]
                else:
                    loops.append(loop_actual)
                    loop_actual = []

        loop_actual.append(word)

        cantidad_palabras = len(loop_actual)
        duracion = loop_actual[-1].end - loop_actual[0].start
        termina_oracion = word.word.strip().endswith(('.', '?', '!'))
        es_ultima = (i == total_palabras - 1)

        if es_ultima:
            loops.append(loop_actual)
            break

        cortar = False

        # A. Fin de oración con al menos aj.min_palabras
        if termina_oracion and cantidad_palabras >= aj.min_palabras:
            cortar = True

        # B. Tope duro de palabras o tiempo
        if not cortar and (cantidad_palabras >= aj.max_palabras or duracion >= aj.max_segundos):
            # Buscar atrás la última coma o fin de oración para cortar limpio
            corte_atras = -1
            for j in range(len(loop_actual) - 1, aj.min_palabras - 1, -1):
                if loop_actual[j].word.strip().endswith(('.', '?', '!', ',')):
                    corte_atras = j
                    break

            if corte_atras >= 0:
                # Cortar en la coma/punto y pasar el resto al siguiente bloque
                loops.append(loop_actual[:corte_atras + 1])
                loop_actual = loop_actual[corte_atras + 1:]
                i += 1
                continue
            else:
                # Sin coma ni punto atrás, buscar adelante fin de oración
                palabras_extra = 0
                for j in range(1, aj.tolerancia_adelante + 1):
                    if i + j >= total_palabras:
                        break
                    w_adelante = palabras[i + j]
                    if w_adelante.word.strip().endswith(('.', '?', '!')):
                        for k in range(1, j + 1):
                            loop_actual.append(palabras[i + k])
                        palabras_extra = j
                        break
                i += palabras_extra
                cortar = True

        if cortar:
            loops.append(loop_actual)
            loop_actual = []

        i += 1

    return loops


def segundos_a_srt(segundos):
    """Convierte los segundos al formato de tiempo que usan los archivos SRT"""
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos - int(segundos)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def armar_por_hablante(palabras, hablantes, aj=None):
    """Como armar(), pero un loop nunca mezcla hablantes. Devuelve [(hablante, loop)]."""
    resultado, tramo, actual = [], [], None
    for w, h in zip(palabras, hablantes):
        if tramo and h != actual:
            resultado += [(actual, loop) for loop in armar(tramo, aj)]
            tramo = []
        tramo.append(w)
        actual = h
    if tramo:
        resultado += [(actual, loop) for loop in armar(tramo, aj)]
    return resultado


PAUSA_NOMBRE = 2.0  # segundos: tras una pausa así se repite el nombre aunque hable el mismo


def nombres_por_loop(con_hablante, nombres, todos=True):
    """Prefijo "Nombre" de cada loop: en todos, o solo cuando cambia el hablante o tras una pausa larga.

    nombres: {hablante: nombre}; los que no tienen nombre salen como "Hablante N".
    Si hay un solo hablante y no tiene nombre, no se pone ninguno.
    """
    distintos = {h for h, _ in con_hablante}
    if len(distintos) <= 1 and not any(nombres.get(h) for h in distintos):
        return [None] * len(con_hablante)
    prefijos, anterior, fin_anterior = [], None, None
    for h, loop in con_hablante:
        if todos or h != anterior or loop[0].start - fin_anterior > PAUSA_NOMBRE:
            prefijos.append(nombres.get(h) or f"Hablante {h[1:]}")
        else:
            prefijos.append(None)
        anterior, fin_anterior = h, loop[-1].end
    return prefijos


def escribir_srt(loops, path, prefijos=None):
    with open(path, "w", encoding="utf-8") as f:
        for i, loop in enumerate(loops):
            texto = "".join(w.word for w in loop).strip()
            if prefijos and prefijos[i]:
                texto = f"{prefijos[i]}: {texto}"
            f.write(f"{i + 1}\n")
            f.write(f"{segundos_a_srt(loop[0].start)} --> {segundos_a_srt(loop[-1].end)}\n")
            f.write(f"{texto}\n\n")


def _a_segundos(t):
    h, m, s = t.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def resumen_srt(texto):
    """Una línea con cuántos loops hay y qué tan largos son."""
    duraciones, palabras = [], []
    for bloque in re.split(r"\n\s*\n", texto.strip()):
        lineas = bloque.splitlines()
        if len(lineas) >= 3 and "-->" in lineas[1]:
            inicio, fin = lineas[1].split("-->")
            duraciones.append(_a_segundos(fin) - _a_segundos(inicio))
            palabras.append(len(" ".join(lineas[2:]).split()))
    if not duraciones:
        return "El SRT está vacío."
    n = len(duraciones)
    coma = lambda x: f"{x:.1f}".replace(".", ",")
    return (f"**{n} loops** · promedio {coma(sum(palabras) / n)} palabras y {coma(sum(duraciones) / n)} s · "
            f"el más largo {coma(max(duraciones))} s · el de más palabras {max(palabras)}")
