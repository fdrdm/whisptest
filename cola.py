"""Cola única de trabajos: transcripción, traducción y separación en orden de llegada.

Un solo hilo trabajador procesa de a un trabajo, así que la GPU nunca tiene dos
trabajos a la vez. Los trabajos siguen aunque quien los mandó cierre el navegador:
los resultados quedan en output_* y se pueden bajar desde la pestaña Cola.
Un trabajo en espera se puede sacar de la cola; uno que ya corre se detiene en su
próximo aviso de progreso (cada segmento, bloque o lote).
La cola vive en memoria: reiniciar el servicio pierde los trabajos en espera.
"""
import itertools
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import gradio as gr

import gpu

MAX_HISTORIAL = 100


class Cancelado(Exception):
    pass


@dataclass
class Trabajo:
    id: int
    tipo: str
    archivo: str
    fn: Callable
    args: tuple
    ip: str
    estado: str = "en espera"  # en espera / procesando / listo / error / cancelado
    progreso: float = 0.0
    detalle: str = ""
    resultado: Any = None
    error: str = ""
    creado: float = field(default_factory=time.time)
    inicio: float | None = None
    fin: float | None = None
    hecho: threading.Event = field(default_factory=threading.Event)
    cancelar_pedido: bool = False

    def reportar(self, fraccion, desc=""):
        if self.cancelar_pedido:
            raise Cancelado()
        self.progreso = fraccion
        self.detalle = desc

    def archivos(self):
        """Rutas de archivos existentes dentro del resultado."""
        valores = self.resultado if isinstance(self.resultado, (tuple, list)) else (self.resultado,)
        return [v for v in valores if isinstance(v, str) and os.path.isfile(v)]


_ids = itertools.count(1)
_pendientes = queue.Queue()
_trabajos = []
_lock = threading.Lock()


def _trabajador():
    while True:
        t = _pendientes.get()
        with _lock:
            if t.estado == "cancelado":  # lo sacaron mientras esperaba
                continue
            t.estado = "procesando"
        t.inicio = time.time()
        try:
            t.resultado = t.fn(*t.args, ip=t.ip, progress=t.reportar)
            t.estado = "listo"
        except Cancelado:
            t.estado = "cancelado"
            logging.info(f"CANCELADO\t{t.tipo}\t{t.archivo}\tdurante el proceso\tip={t.ip}")
        except Exception as e:
            t.error = getattr(e, "message", None) or str(e) or type(e).__name__
            t.estado = "error"
            logging.info(f"ERROR\t{t.tipo}\t{t.archivo}\t{t.error}\tip={t.ip}")
        finally:
            # Recién acá el trabajo soltó sus referencias (variables locales, traceback),
            # así que es donde la VRAM vuelve a quedar libre de verdad
            gpu.liberar()
            t.fin = time.time()
            t.hecho.set()


threading.Thread(target=_trabajador, daemon=True, name="cola-gpu").start()


def ip(request):
    return request.client.host if request else "?"


def encolar(tipo, archivo, fn, args, ip):
    t = Trabajo(next(_ids), tipo, archivo, fn, args, ip)
    with _lock:
        _trabajos.append(t)
        terminados = [x for x in _trabajos if x.hecho.is_set()]
        for viejo in terminados[:max(0, len(_trabajos) - MAX_HISTORIAL)]:
            _trabajos.remove(viejo)
    _pendientes.put(t)
    return t


def _en_espera():
    with _lock:
        return [t for t in _trabajos if t.estado == "en espera"]


def posicion(t):
    return _en_espera().index(t) + 1 if t.estado == "en espera" else 0


def cancelar(id_trabajo):
    """Saca un trabajo en espera o pide detener uno que está corriendo. Devuelve un mensaje."""
    with _lock:
        t = next((x for x in _trabajos if x.id == id_trabajo), None)
        if not t:
            return "Elegí un trabajo de la tabla."
        if t.hecho.is_set():
            return f"El trabajo #{t.id} ya terminó ({t.estado})."
        if t.estado == "en espera":
            t.estado = "cancelado"
            t.fin = time.time()
            t.hecho.set()
            logging.info(f"CANCELADO\t{t.tipo}\t{t.archivo}\ten espera\tip={t.ip}")
            return f"Trabajo #{t.id} sacado de la cola."
        t.cancelar_pedido = True
    return f"Cancelando el trabajo #{t.id}: se detiene en el próximo paso."


def esperar(t, progress):
    """Mantiene al navegador al tanto mientras el trabajo espera o corre y devuelve el resultado."""
    while not t.hecho.wait(1):
        if t.estado == "en espera":
            progress(0, desc=f"En cola: posición {posicion(t)} (trabajo #{t.id}). "
                             "Podés cerrar la página: el resultado queda en la pestaña Cola.")
        else:
            progress(t.progreso, desc=t.detalle or "Procesando...")
    if t.estado == "error":
        raise gr.Error(t.error)
    if t.estado == "cancelado":
        raise gr.Error(f"El trabajo #{t.id} fue cancelado.")
    return t.resultado


def _duracion(segundos):
    m, s = divmod(int(segundos), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def tabla():
    ahora = time.time()
    with _lock:
        trabajos = list(reversed(_trabajos))
    filas = []
    for t in trabajos:
        if t.estado == "en espera":
            estado = f"en espera ({posicion(t)}°)"
        elif t.estado == "procesando":
            estado = "cancelando..." if t.cancelar_pedido else f"procesando {t.progreso:.0%}"
        elif t.estado == "error":
            estado = f"error: {t.error}"
        elif t.estado == "cancelado":
            estado = "cancelado"
        else:
            estado = "listo"
        espera = _duracion((t.inicio or t.fin or ahora) - t.creado)
        duracion = _duracion((t.fin or ahora) - t.inicio) if t.inicio else ""
        filas.append([t.id, t.tipo, t.archivo, estado, time.strftime("%H:%M:%S", time.localtime(t.creado)), espera, duracion])
    return filas


def resumen_html():
    with _lock:
        actual = next((t for t in _trabajos if t.estado == "procesando"), None)
    esperando = len(_en_espera())
    if not actual and not esperando:
        return "<div>Cola: vacía</div>"
    texto = f"procesando #{actual.id} ({actual.tipo}, {actual.progreso:.0%})" if actual else "arrancando"
    return f"<div>Cola: {texto} · {esperando} en espera</div>"


def build_tab():
    with gr.Tab("Cola"):
        gr.Markdown(
            "Todos los trabajos (transcribir, traducir y separar) van a una sola cola y se procesan "
            "de a uno, en orden de llegada. Siguen aunque cierres la página. "
            "Hacé clic en un trabajo para elegirlo: si terminó, abajo aparecen sus resultados; "
            "si está en espera o procesando, lo podés cancelar (por ejemplo, si lo mandaste dos veces)."
        )
        tabla_cola = gr.Dataframe(
            headers=["#", "Tipo", "Archivo", "Estado", "Enviado", "Espera", "Duración"],
            value=tabla,
            interactive=False,
            wrap=True,
            max_height=500,
        )
        seleccionado = gr.State(None)
        with gr.Row():
            info = gr.Markdown("Ningún trabajo elegido.")
            btn_cancelar = gr.Button("Cancelar trabajo elegido", variant="stop", scale=0)
        resultados = gr.File(label="Resultados del trabajo elegido", file_count="multiple", interactive=False)

        def on_select(evt: gr.SelectData):
            id_trabajo = evt.row_value[0] if evt.row_value else None
            with _lock:
                t = next((x for x in _trabajos if x.id == id_trabajo), None)
            if not t:
                return None, None, "Ningún trabajo elegido."
            archivos = t.archivos() if t.estado == "listo" else None
            return archivos, t.id, f"Elegido: **#{t.id}** {t.tipo} · {t.archivo} · {t.estado}"

        def on_cancelar(id_trabajo):
            return cancelar(id_trabajo), tabla()

        tabla_cola.select(fn=on_select, outputs=[resultados, seleccionado, info])
        btn_cancelar.click(fn=on_cancelar, inputs=[seleccionado], outputs=[info, tabla_cola])
        gr.Timer(3).tick(fn=tabla, outputs=[tabla_cola])
