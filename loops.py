"""Armado de loops (bloques del SRT) a partir de las palabras con tiempos de Whisper.

Las palabras de cada transcripción se guardan en JSON (output_translate/palabras/),
así los loops se pueden rearmar con otros ajustes sin volver a transcribir.
"""
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


def escribir_srt(loops, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, loop in enumerate(loops):
            f.write(f"{i + 1}\n")
            f.write(f"{segundos_a_srt(loop[0].start)} --> {segundos_a_srt(loop[-1].end)}\n")
            f.write(f"{''.join(w.word for w in loop).strip()}\n\n")
