# Whisptest

Este proyecto permite la transcripción, separación y traducción de subtítulos mediante Whisper, Demucs y Ollama local.

## 🚀 Instalación rápida (Entorno Virtual + Ollama)

Para clonar el proyecto e instalar todas las dependencias y modelos en un solo comando:

```bash
git clone git@github.com:fdrdm/whisptest.git
cd whisptest
./setup.sh
```

El script `./setup.sh` se encargará de:
1. Crear el entorno virtual (`.venv`).
2. Instalar todas las librerías de Python desde `requirements.txt`.
3. Verificar/Instalar Ollama y descargar los modelos locales necesarios (`translategemma:latest`).

## 🦙 Ollama Local

El servidor de traducción utiliza Ollama en modo local (`http://localhost:11434/api/generate`). Asegúrate de tener Ollama corriendo antes de iniciar la traducción:

```bash
ollama serve
```

Y tener descargados los modelos:
```bash
ollama pull translategemma:latest
```

## 💡 Ejecutar la app

Todo corre en un solo proceso (`app.py`, puerto 8188, publicado en [ia.fdrdm.com](https://ia.fdrdm.com)) con pestañas para transcribir, traducir SRT, separar audio, ver la cola, los archivos y el historial.

Como servicio de usuario (recomendado):

```bash
systemctl --user enable --now whisptest
journalctl --user -u whisptest   # o tail -f app.log
```

O a mano:

```bash
source .venv/bin/activate
python3 app.py
```

## 🔁 Loops (`loops.py`)

Cada transcripción guarda sus palabras con tiempos en `output_translate/palabras/<video>_<idioma>.json`. En la pestaña **Transcribir**, el desplegable *Ajustes de loops* (máx. palabras, máx. segundos, mínimo para cortar en un punto y pausa para el corte suave) se usa al transcribir y para **Rearmar con estos ajustes**: reescribe el SRT al instante desde ese JSON, sin GPU ni cola. Las transcripciones anteriores a esto no tienen JSON y hay que volver a transcribirlas una vez. El corte suave solo conoce conjunciones en inglés, y los topes pueden pasarse hasta 5 palabras para cerrar la oración.

## 🗣️ Hablantes (`hablantes.py`)

Con **Identificar hablantes** (activado por defecto) el trabajo de transcripción corre pyannote (`pyannote/speaker-diarization-community-1`) después de Whisper: unos 15 s más y 0,6 GB de VRAM (lotes de 8). Cada palabra queda asignada a un hablante en `palabras/<video>_<idioma>.hablantes.json`, con una muestra de audio de cada uno. Los loops nunca mezclan hablantes y el SRT lleva `Nombre: texto` (en todos los loops, o solo cuando cambia el hablante o tras una pausa de más de 2 s). En **Resultado → Hablantes** se escuchan las muestras y se ponen los nombres (sugiere los usados en otros videos). Al traducir, los nombres se separan antes de mandar el texto al modelo y se vuelven a poner después.

Requiere aceptar los términos del modelo en Hugging Face y un token en `~/.cache/huggingface/token` solo para la primera descarga.

## 🎙️ Nemotron (`nemotron.py`)

En **Transcribir → Motor** se puede elegir NVIDIA Nemotron 3.5 Transcribe en vez de Whisper. Tarda más (unos 2 min cada 10 de audio; hasta 1 hora por archivo), pero detecta solo los hablantes (no usa pyannote) y suele separar mejor quién dice cada frase. Nemotron no da tiempos: su texto se alinea con Whisper (`model.align`, ~30 s cada 10 min, ~10 ms de diferencia con los tiempos de Whisper) y desde ahí todo es igual: JSON de palabras, hablantes, loops y **Rearmar**.

- Corre local con `llama-server` de llama.cpp, que se levanta por trabajo en el puerto 8091 y se baja al terminar (carga en ~5 s). Los expertos del MoE van a RAM (~10–12 GB) y el resto a VRAM (~3,6 GB). El servidor corre dentro de un scope de systemd con tope de 30 GB de RAM: si se pasa, se mata solo ese proceso y no se cuelga la PC. Su salida va a `nemotron.log`.
- No hay paquete de pip que sirva todavía (`llama-cpp-python` no trae el audio de Nemotron), así que `setup.sh` compila llama.cpp (commit fijado) en `~/llama.cpp` y baja el modelo IQ4_NL + `mmproj` F16 (~20 GB) de `maternion/Nemotron-3.5-Transcribe-30B-A4B-IQ-GGUF` a `~/modelos/nemotron-transcribe`. Otras rutas: `NEMOTRON_LLAMA_SERVER` y `NEMOTRON_DIR`.
- Usa los prompts oficiales del model card, con el texto antes del audio (con otro prompt el modelo resume o se queda en bucle en audios largos).

## 🎛️ Manejo de VRAM (`gpu.py`)

- Una sola cola de trabajos (`cola.py`): transcripción, traducción y separación se procesan de a una, en orden de llegada, y siguen aunque quien los mandó cierre la página. La pestaña **Cola** muestra todos los trabajos y deja bajar los resultados de los terminados o cancelar uno (en espera sale de la cola; si ya corre, se detiene en el próximo segmento, bloque o lote). La cola vive en memoria: reiniciar el servicio pierde los trabajos en espera.
- Los modelos se cargan por trabajo y se descargan apenas termina (Whisper carga en ~10 s, Demucs en ~2 s); al terminar de traducir también se saca `translategemma` de Ollama.
- PyTorch queda limitado a `WHISPTEST_VRAM_GB` (5 GB por defecto, se cambia en `~/.config/systemd/user/whisptest.service`). Si un trabajo lo supera, falla con un mensaje claro en vez de dejar sin memoria al escritorio.
- Antes de empezar se verifica la VRAM libre; si no alcanza (por ejemplo, porque Ollama tiene un modelo grande cargado) se avisa qué la está usando.
- `uso.log` registra `vram_pico` de cada trabajo para calibrar `NECESARIO_GB` en `gpu.py`.
