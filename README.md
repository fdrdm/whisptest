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

Cada transcripción guarda sus palabras con tiempos en `output_translate/palabras/<video>_<idioma>.<motor>.json`. En la pestaña **Transcribir** se pueden subir varios archivos a la vez (un trabajo de la cola por archivo). El desplegable *Ajustes de loops* (máx. palabras, máx. segundos, mínimo para cortar en un punto y pausa para el corte suave) se usa al transcribir y para **Rearmar con estos ajustes**: reescribe el SRT al instante desde ese JSON, sin GPU ni cola. Las transcripciones anteriores a esto no tienen JSON y hay que volver a transcribirlas una vez. El corte suave solo conoce conjunciones en inglés, y los topes pueden pasarse hasta 5 palabras para cerrar la oración.

## 🗣️ Hablantes (`hablantes.py`)

Cada transcripción es un proyecto con las dos versiones: siempre se identifican los hablantes y los nombres se ponen después. Con Whisper el trabajo corre pyannote (`pyannote/speaker-diarization-community-1`) a continuación: unos 15–40 s más y 0,6 GB de VRAM (lotes de 8); Nemotron ya trae los hablantes. Cada palabra queda asignada a un hablante en `palabras/<video>_<idioma>.<motor>.hablantes.json`, con una muestra de audio de cada uno. Los loops nunca mezclan hablantes y el `.nombres.srt` lleva `Nombre: texto` (en todos los loops, o solo cuando cambia el hablante o tras una pausa de más de 2 s). En **Resultado → Hablantes** se escuchan las muestras y se ponen los nombres (sugiere los usados en otros videos). Al traducir, los nombres se separan antes de mandar el texto al modelo y se vuelven a poner después.

Los archivos llevan el motor en el nombre, así se pueden comparar: `<video>_<idioma>.whisper.srt` / `.nemotron.srt` / `.deepinfra.srt` (limpio, sin hablantes) y, con hablantes, `<video>_<idioma>.<motor>.nombres.srt` con los mismos loops y `Nombre: texto` (los que no tienen nombre salen como "Hablante N"). Las traducciones conservan los sufijos: `<video>_es.nemotron.nombres.srt`. Los archivos de antes (sin motor en el nombre) siguen funcionando.

### Editor de hablantes (`editor.py`, `/editor`)

Página aparte (se abre desde *Resultado → Hablantes*) para corregir quién dice cada loop: el video original se abre desde la compu de quien edita (no se sube, se reproduce desde el disco), el loop que suena se resalta y al hacer clic en uno el video salta ahí. Cada loop muestra inicio y fin y tiene su hablante (o uno nuevo), con atajos `1`–`9`, `↑`/`↓`, `Espacio` y `U` (unir con el siguiente). Velocidad 1x/1.5x/2x y filtro por hablante o "Sin nombre", que puede reproducir solo esas líneas (para reconocer a alguien y ponerle nombre). El texto se corrige haciendo clic (vacío = se borra el loop): con la misma cantidad de palabras cada una conserva sus tiempos, si no se reparten dentro del loop.

Al guardar, los hablantes van al `.hablantes.json` y el texto corregido a las palabras del `.json` (un segmento por loop), así *Rearmar* respeta las dos cosas; las uniones de loops no (Rearmar arma los loops de cero). Se reescriben los dos SRT.

Requiere aceptar los términos del modelo en Hugging Face y un token en `~/.cache/huggingface/token` solo para la primera descarga.

## 🎙️ Nemotron (`nemotron.py`)

En **Transcribir → Motor** se puede elegir NVIDIA Nemotron 3.5 Transcribe en vez de Whisper. Tarda más (unos 2 min cada 10 de audio), pero detecta solo los hablantes (no usa pyannote) y suele separar mejor quién dice cada frase. Nemotron no da tiempos: su texto se alinea con Whisper (`model.align`, ~30 s cada 10 min, ~10 ms de diferencia con los tiempos de Whisper) y desde ahí todo es igual: JSON de palabras, hablantes, loops y **Rearmar**.

El audio va en tramos de ~5 min (`nemotron.TRAMO_SEG`, los mismos que usa DeepInfra), cortados en el medio segundo más silencioso del último minuto de cada tramo (así no se parte ninguna palabra). Es más estable que mandar todo junto y no tiene límite de duración. Los hablantes de cada tramo son distintos: al mismo personaje se le pone el mismo nombre en cada uno y en el `.nombres.srt` queda igual.

- Corre local con `llama-server` de llama.cpp, que se levanta por trabajo en el puerto 8091 y se baja al terminar (carga en ~5 s). Los expertos del MoE van a RAM (~10–12 GB) y el resto a VRAM (~3,3 GB). El servidor corre dentro de un scope de systemd con tope de 30 GB de RAM: si se pasa, se mata solo ese proceso y no se cuelga la PC. Su salida va a `nemotron.log`.
- No hay paquete de pip que sirva todavía (`llama-cpp-python` no trae el audio de Nemotron), así que `setup.sh` compila llama.cpp (commit fijado) en `~/llama.cpp` y baja el modelo IQ4_NL + `mmproj` F16 (~20 GB) de `maternion/Nemotron-3.5-Transcribe-30B-A4B-IQ-GGUF` a `~/modelos/nemotron-transcribe`. Otras rutas: `NEMOTRON_LLAMA_SERVER` y `NEMOTRON_DIR`.
- Usa los prompts oficiales del model card, con el texto antes del audio (con otro prompt el modelo resume o se queda en bucle en audios largos).
- Si se queda repitiendo (por ejemplo, el "la la la…" de una canción en Daria 101), lo detecta mientras genera, corta y reintenta con la penalización de repeticiones DRY de llama.cpp. DRY no se usa siempre porque también toca repeticiones legítimas ("self-esteem" → "self esteem"). Queda en `uso.log` como `NEMOTRON se quedó repitiendo`.

### Nemotron en DeepInfra

**Motor → Nemotron (DeepInfra)** manda el audio (opus mono de 16 kHz, ~4 MB cada 20 min) al modelo completo (BF16) en [DeepInfra](https://deepinfra.com/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), que ya devuelve cada palabra con sus tiempos y su hablante. El audio va en los mismos tramos de ~5 min que el local, mandados en paralelo: unos 30 s cada 20 de audio y unos USD 0,04 (queda en `uso.log` como `costo_usd` y en el Historial). No corre nada local: el idioma lo detecta el Whisper de DeepInfra (`whisper-large-v3-turbo`) con 30 s desde la mitad de lo hablado (~USD 0,0001).

- La clave va en `~/.config/whisptest.env` (`DEEPINFRA_API_KEY=...`), que el servicio lee con `EnvironmentFile`. Nunca en el repo.
- Se pide siempre con `diarize=true` y la instrucción con `<spk:N>`: sin eso devuelve una sola voz o pierde palabras.
- Por qué en tramos: su alineación de tiempos se degrada con el largo del audio. Daria 102 entera (22 min) vino con 2 min y medio de palabras amontonadas y dos personajes intercalados; el mismo tramo mandado solo vino perfecto (99,8% de las palabras en su tiempo). Como cada pedido numera los hablantes de cero, los hablantes de cada tramo son distintos, igual que en el local.
- Aun así pierde algunas líneas cortas en escenas con voces encimadas que el local sí saca: es su `diarize`, que descarta palabras que no logra asignar a una voz.
- Si aun así los tiempos vienen rotos (amontonados o desordenados; con tramos de 5 min no volvió a pasar), el texto se alinea con Whisper local como con el Nemotron local (es lo único que usa la GPU) (`alineado=whisper` en `uso.log`).

## 🎛️ Manejo de VRAM (`gpu.py`)

- Una sola cola de trabajos (`cola.py`): transcripción, traducción y separación se procesan de a una, en orden de llegada, y siguen aunque quien los mandó cierre la página. La pestaña **Cola** muestra todos los trabajos y deja bajar los resultados de los terminados o cancelar uno (en espera sale de la cola; si ya corre, se detiene en el próximo segmento, bloque o lote). La cola vive en memoria: reiniciar el servicio pierde los trabajos en espera.
- Los modelos se cargan por trabajo y se descargan apenas termina (Whisper carga en ~10 s, Demucs en ~2 s); al terminar de traducir también se saca `translategemma` de Ollama.
- PyTorch queda limitado a `WHISPTEST_VRAM_GB` (5 GB por defecto, se cambia en `~/.config/systemd/user/whisptest.service`). Si un trabajo lo supera, falla con un mensaje claro en vez de dejar sin memoria al escritorio.
- Antes de empezar se verifica la VRAM libre; si no alcanza (por ejemplo, porque Ollama tiene un modelo grande cargado) se avisa qué la está usando.
- `uso.log` registra `vram_pico` de cada trabajo para calibrar `NECESARIO_GB` en `gpu.py`.
