# Informe para agentes: motores de transcripción, hablantes y editor

Estado al 2026-11-06 (noche). Este archivo resume lo que se hizo, qué funcionó y qué no, para que otro agente (Claude, Codex) siga sin repetir pruebas. La arquitectura general está en `README.md` y en `/home/fede/CLAUDE.md`. El usuario trabaja en español rioplatense, quiere cambios simples y que le pregunten antes de mandar trabajos pesados o pagos.

## 1. Qué hay hoy en la app

App Gradio única (`app.py`, puerto 8188, servicio `systemctl --user restart whisptest`), con una sola cola en memoria (`cola.py`). La pestaña **Transcribir** tiene tres motores (`server_translate.MOTORES`):

| Motor (UI) | Sufijo de archivos | Cómo funciona |
|---|---|---|
| Whisper | `.whisper` | stable-ts `turbo` transcribe; pyannote `speaker-diarization-community-1` pone hablantes (`hablantes.py`). |
| Nemotron | `.nemotron` | `nemotron.py`: llama-server de llama.cpp con el GGUF IQ4_NL local, en tramos de ~5 min. Da el texto con etiquetas `<spk:N>`; los tiempos salen de `model.align(vad=True)` de Whisper (`server_translate._alinear`). |
| Nemotron (DeepInfra) | `.deepinfra` | `nemotron.deepinfra()`: API de DeepInfra con el modelo BF16, los mismos tramos de ~5 min mandados en paralelo. Devuelve palabras con tiempo y hablante; no corre nada local salvo el plan B. |

Comportamiento común:

- **Varios archivos a la vez.** Un trabajo de la cola por archivo; si uno falla, los demás siguen.
- **Hablantes siempre.** Cada transcripción guarda el SRT limpio (`<video>_<idioma>.<motor>.srt`), el `.nombres.srt` (con `Hablante N:` o el nombre), las palabras con tiempos en `palabras/<...>.json` y los hablantes en `palabras/<...>.hablantes.json` con una muestra de audio de cada uno.
- **Editor** (`editor.py` + `static/editor.html`, ruta `/editor`): abre el video desde el disco del usuario, cambia el hablante por loop, corrige texto, une loops, velocidad 1x/1.5x/2x, filtro por hablante y **combinar dos hablantes en uno** (al pie del panel de Hablantes).
- **Interfaz:** estilo "bento" con el verde de wider-media, tarjeta de estado de la cola con un punto que late, botones que dicen "Agregar a la cola (N trabajos antes)", resultados vacíos ocultos.

### Nemotron local: detalles que importan

- **Servidor.** `~/llama.cpp/build/bin/llama-server`, compilado en el commit `e14a779` (ver `setup.sh`). Opciones: `--cpu-moe -ngl 99 -c 16384 -np 1 --reasoning off`, puerto 8091, dentro de `systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0` (una vez colgó la PC por falta de RAM).
- **Modelo.** `maternion/Nemotron-3.5-Transcribe-30B-A4B-IQ-GGUF` (IQ4_NL + mmproj F16) en `~/modelos/nemotron-transcribe`. ~3,3 GB de VRAM y ~18,6 GB de RSS (casi todo el modelo mapeado; no baja con menos contexto). Se levanta por trabajo y se baja al terminar.
- **Tramos.** `nemotron.tramos(audio)` corta cada `TRAMO_SEG = 300` s en el medio segundo más silencioso del último minuto. Un tramo de 5 min usa ~3700 tokens de audio + ~1000 de texto. Los hablantes de cada tramo se prefijan con el número de tramo (`"0:1"`, `"1:1"`): **se numeran de nuevo en cada tramo**. Antes eran de 10 min; se bajó a 5 para que coincidan con DeepInfra (pedido del usuario).
- **Prompt.** Texto antes del audio, temperatura 0, razonamiento apagado, el prompt oficial "Who said what, and when": `Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.`
- **Bucles.** Si en las últimas 150 palabras hay menos del 20% de 6-gramas distintos, o `finish_reason == length`, reintenta una vez con DRY (`dry_multiplier 0.8, dry_base 1.75, dry_allowed_length 2`).
- **Alineación.** `model.align(..., vad=True)`: la voz la detecta Silero. Sin VAD, una intro musical estira el texto (Futurama quedaba 40 s corrido).

### Nemotron en DeepInfra: detalles que importan

- **Clave.** `DEEPINFRA_API_KEY` en `~/.config/whisptest.env` (permisos 600), cargada por el servicio con `EnvironmentFile=-%h/.config/whisptest.env`. **Nunca va en el repo.** `Nemotron_DeepInfra_*.md` tiene la clave y está en `.gitignore`.
- **Endpoint.** `POST https://api.deepinfra.com/v1/inference/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`, multipart: `audio` (opus mono 16 kHz 32k, generado con ffmpeg desde las muestras del tramo), `chunk_level=word`, `diarize=true`, `instruction` = el mismo prompt. Es la mejor instrucción de las 5 probadas (ver sección 3).
- **Tramos en paralelo.** `nemotron.deepinfra(audio, cantidad, progress)` manda los tramos de `tramos()` con un `ThreadPoolExecutor(4)`, corre los tiempos y prefija el hablante con el tramo. Un capítulo de 22 min tarda ~26–36 s en total.
- **Idioma.** Lo da el Whisper de DeepInfra (`openai/whisper-large-v3-turbo`) sobre 30 s desde la palabra del medio (~USD 0,0001), porque la respuesta de Nemotron trae `language: ""`.
- **Control de tiempos** (`nemotron.tiempos_sanos`): rota si más del 0,5% de las palabras retrocede más de 0,5 s o hay 80 palabras en 10 s. **Plan B:** realinear con Whisper local (no arregla texto ya intercalado; con tramos de 5 min no volvió a saltar).
- **Respuesta cruda** siempre en `palabras/<...>.deepinfra.crudo.json` (excluida de la lista de transcripciones).
- **Costo.** USD 0,002 por minuto de audio: ~USD 0,045 por capítulo de 22 min.

## 2. Resultados medidos

### Herramientas (en `herramientas/`)

- `comparar_motores.py REF.srt BASE [motor=archivo.json ...]`: WER contra subtítulos de referencia, **con medición justa**: saca muletillas de los dos lados, unifica gonna/going to y números, trata "|" del OCR como "I" y no cuenta diferencias de espacios. Sin esto los subtítulos (que resumen) castigan a los motores literales.
- `consenso_motores.py nombre=archivo.json ...`: sin referencia. Acuerdo de a pares y "errores propios" (tramos donde un motor difiere y los otros dos coinciden).
- `medir_sin_referencia.py BASE`: sin referencia, todo junto: consenso, sincronía entre motores (±0,3 s), cambios de hablante a mitad de frase, loops de menos de 1 s, e identidad contra la voz de pyannote de la corrida de Whisper (un hablante = una voz / una voz = un hablante).

Referencias usadas: Futurama S01E03 (subtítulos para sordos del DVD, en bitmap, pasados por OCR: buena), Daria 102 (subtítulos del mkv: con errores, sirve para comparar entre motores), Jancy Family "Dream Bedroom" (YouTube, 21 min: solo captura automática, **no se usó como referencia**; se midió por consenso).

### Texto, medición justa (WER; más bajo es mejor)

| | Whisper | Nemotron local | DeepInfra |
|---|---|---|---|
| Futurama | 6,7% | **6,2%** | 11,5% (entero) |
| Futurama: palabras perdidas | 53 | **29** | 89 |
| Futurama: errores que solo tiene ese motor | 21 | **12** | 18 |
| Daria 102 | **10,7%** | 11,6% | 25,9% entero → 16,2% en tramos |
| Daria 102: errores propios | 22 | 23 | 38 |
| Jancy (consenso): tramos donde es el único que difiere | 237 | **184** | 286 |

**Conclusión sobre el texto:** Whisper y Nemotron local empatan (cada uno gana un video por poco; en Jancy el local es el que más coincide con los otros dos). Una conclusión anterior ("Whisper escribe mejor") era un defecto de la medición sin normalizar. DeepInfra escribe igual que el local (mismas sustituciones) pero lo que entrega pierde más palabras: en escenas con voces encimadas su `diarize` descarta líneas cortas ("You're a great hostess, cupcake", "Let's be friends") que el local sí saca.

### Sincronía y hablantes

| | Whisper | Nemotron local | DeepInfra |
|---|---|---|---|
| Futurama: palabras en su tiempo | 98,0% | 95,7% (con VAD; 87,9% sin) | 97,5% |
| Daria 102: palabras en su tiempo | 98,4% | 98,1% | 93,9% entero (72% en 7:20–10:00) → **99,8% en tramos** |
| Futurama: líneas de una persona con un solo hablante | 95,4% | **100%** | 98,9% |
| Daria 102: ídem | 90,9% | **99,7%** | 95,5% |
| Futurama: cada hablante es una sola voz (pyannote) | ref | 68% | 72% |
| Jancy: cambios de hablante a mitad de frase | 25% | **18%** | 58% |
| Jancy: loops / de menos de 1 s | 716 / 366 | 676 / **324** | 1057 / **703** |
| Jancy: hablantes · una voz = un hablante | 4 · ref | 21 · 24% | 22 · 23% |

### Tiempos y costo (capítulos de 21–22 min)

| | Whisper | Nemotron local | DeepInfra entero | DeepInfra en tramos |
|---|---|---|---|---|
| Daria 102 | 96 s | 291 s (10 min) | 108 s (con plan B) | 26 s (directo) |
| Futurama | 119 s | 378 s (10 min) | 33 s | – |
| Jancy | 91 s | 349 s (10 min) / 376 s (5 min) | 36 s | 36 s |
| Costo | – | – | USD 0,043–0,045 | igual |
| Recursos | 3,5 GB VRAM | 3,3 GB VRAM + 19 GB RAM | nada local | nada local |

### Puntajes de Jancy (1 = una cagada, 5 = bárbaro), sin YouTube

| Categoría | Whisper + pyannote | Nemotron local | DeepInfra (tramos) |
|---|---|---|---|
| Velocidad | 4 | 2 | **5** |
| Texto (consenso) | 3 | **4** | 2 |
| Sincronía | 4 | 4 | 4 |
| Turnos (dónde cambia quién habla) | 3 | **5** | 2 |
| Legibilidad de los loops | 3 | **4** | 1 |
| Identidad (misma persona = mismo número) | **4** | 2 | 2 |
| Confiabilidad | **5** | 4 | 4 |
| Costo y recursos | **5** | 3 | 4 |
| **Promedio** | **3,9** | **3,5** | **3,0** |

La identidad baja en los dos Nemotron por los tramos de 5 min (numeran de cero en cada uno); el arreglo pendiente es el híbrido con pyannote (sección 4). Hay un informe HTML para humanos en `~/Downloads/informe_motores_transcripcion.html`.

## 3. Qué funcionó y qué no

### Funcionó

- **Nemotron local estable:** scope de systemd + `-np 1`, `--reasoning off`, prompt oficial con el texto antes del audio, tramos cortados en silencios (sin bucles, sin límite de 1 h), `align(vad=True)`.
- **DeepInfra en tramos de 5 min.** Su alineación de tiempos se degrada con el largo del audio: Daria 102 entera venía con 2,5 min de palabras amontonadas y dos personajes intercalados palabra por palabra; el tramo 7:00–10:30 mandado solo vino con 99,8% de las palabras en su tiempo. Con 5 tramos en paralelo: 26 s, mismo costo, tiempos sanos en 2 de 2 capítulos. **Fue el "dar en el clavo" que pidió el usuario.**
- **Separación de turnos de Nemotron** (local): dentro de una escena separa casi perfecto quién dice cada línea (99,7–100% contra 91–95% de pyannote) y es el que menos corta a mitad de frase.
- **El `instruction` en DeepInfra hace falta:** sin él el WER pasa de 19% a 39%.
- **Híbrido turnos de Nemotron + voz de pyannote** (prototipo, no implementado): en Futurama dejó a Bender, Fry, el Profesor y Leela siempre con el mismo número y bajó de 17 a 11 hablantes.

### No funcionó / descartado

- **Identidad de hablantes en Nemotron:** reusa los números según el rol en cada escena y, con los tramos, numera de cero cada 5 min. DeepInfra tiene el mismo problema aunque no se corte (su `diarize` agrupa voces mal en dibujos animados: `speaker_1` fue Fry, el Profesor y Bender).
- **Corrección:** DeepInfra **no** renumera los hablantes cada 10 min cuando se manda el audio entero (en Daria 102 y Futurama hay hablantes que cruzan el minuto 10; en Daria 101 coincidió un cambio de escena). Las versiones anteriores de este informe lo afirmaban.
- **Instrucciones alternativas en DeepInfra** (todas en Futurama): con "assign one stable tag per speaker…" y puntuación, igual que la actual; "Who said what" de su página sin puntuación, WER 42% y tiempos peores; "Transcribe verbatim. Start every speaker turn…", 5 hablantes mezclados y 58% en tiempo. La instrucción cambia la calidad de la alineación de DeepInfra; la actual es la mejor.
- **DeepInfra con `chunk_level=segment`:** mismo intercalado. **Por la API compatible con OpenAI:** `model_not_found`.
- **Unir hablantes entre tramos por embeddings de pyannote:** no confiable en dibujos (uniones correctas 0,72–0,82; voces distintas 0,78).
- **Plan B de DeepInfra (realinear con Whisper):** no arregla texto ya intercalado; quedó solo por si acaso.
- **Fragmentación de DeepInfra:** cuando dos hablan a la vez intercala las palabras de los dos: en Jancy 703 loops de menos de 1 s. Con tramos más chicos no cambiaría.

### Por qué DeepInfra fallaba con el audio entero

Según el model card de NVIDIA (`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`): el texto lo escribe el modelo de lenguaje; los tiempos por palabra los pone una **cabeza CTC** que alinea ese texto con el audio; los hablantes los relaciona un codificador de voces Sortformer; la respuesta oficial de vLLM trae `ctc_timestamps.error` para descartar alineaciones fallidas. DeepInfra no expone ese `error`, entrega los tiempos corridos (hasta 90 s), ordena por tiempo y deja a dos hablantes intercalados; el campo `text` también sale intercalado. Las palabras de cada hablante, por separado, sí están en orden. La falla depende del largo del audio: con 5 min no apareció.

El GGUF local **no trae la cabeza CTC** (sí el codificador de voces: tensores `a.spk.*` en el mmproj), y llama.cpp no la implementa. Por eso el local usa Whisper para los tiempos. La versión oficial completa necesita vLLM con una GPU de 80 GB.

## 4. Pendientes y decisiones abiertas

1. **Identidad estable de hablantes:** implementar el híbrido (turnos de Nemotron + voz de pyannote) para los dos Nemotron. Suma 15–40 s de pyannote (0,6 GB de VRAM). Es lo que más mejoraría el puntaje de los Nemotron. El usuario no lo aprobó todavía.
2. **Escribirle a DeepInfra** para que exponga `ctc_timestamps.error` y/o el texto crudo con `<spk:N>` en orden.
3. **Probar en el Nemotron local** el agregado "assign one stable tag per speaker and keep speaker identities consistent" (en DeepInfra no sirvió porque allá el número lo pone su `diarize`).
4. **Nombre de la nena de Jancy:** Whisper escribe "Starling", los Nemotron "Starry", YouTube "Starley". Decidir de oído cuál acertó.
5. Ideas: unir loops de una sola palabra que deja Nemotron con habla superpuesta; probar un GGUF Q6 (~25 GB; la PC tiene 46 GB de RAM); identificar hablantes en transcripciones viejas sin ellos; achicar los tramos de DeepInfra si vuelve a fallar.

## 5. Trampas conocidas

- **No editar `uso.log` con `sed -i` ni reemplazarlo con el servicio corriendo:** el servicio sigue escribiendo en el archivo borrado. Para limpiar: `grep -v ... > tmp && cat tmp > uso.log`, o reiniciar el servicio.
- **Los clientes de prueba pueden ensuciar `uso.log`:** importar `server_translate` desde un script configura el logging hacia `uso.log`; para pruebas importar solo `loops`.
- **La API `/procesar_en_cola` recibe una lista de archivos:** `(archivos[], motor, num_hablantes, max_palabras, max_segundos, min_palabras, pausa_suave)`. Un archivo suelto da error de pydantic `list_type`.
- **La cola vive en memoria:** reiniciar el servicio pierde los trabajos en espera.
- **`ffmpeg` a ogg sin `-vn`** recodifica también el video y tarda minutos.
- **Futurama:** los subtítulos son `dvd_subtitle` (bitmap): superponerlos sobre fondo negro con `ffmpeg -f lavfi -i color=black:s=720x480:r=10 -i X.mkv -filter_complex "[0:v][1:s:0]overlay,mpdecimate,showinfo"` y pasar `tesseract --psm 6` sobre la imagen invertida. El OCR lee la "I" como "|".
- **YouTube:** `yt-dlp --list-subs` muestra si hay subtítulos manuales ("Available subtitles") o solo automáticos; los automáticos vienen "rodantes" (cada bloque repite la línea anterior) y hay que deduplicarlos. Varios pedidos seguidos dan 429.
- **Gradio 6:** el tema y el favicon van en `mount_gradio_app`. El tema fuerza oscuro copiando los valores `*_dark` sobre los claros.

## 6. Datos que quedaron

- Videos de prueba: `/run/media/fede/Datos/Daria restoration/daria/daria_10{1,2}_esteemsters/the_invitation.mkv`, `.../Futurama (1999) Season 1-7 .../Season 1/Futurama (1999) - S01E03 - I, Roommate (...).mkv`, `~/Videos/Jancy Family - Dream Bedroom.mp4` (+ `.referencia.srt` deduplicado de YouTube, `.en-orig.srt` original).
- Salidas en `output_translate/` (las tres versiones de Daria 101, Daria 102, Futurama S01E03 y Jancy; Jancy y Daria 102 con el local de 5 min, Jancy con DeepInfra en tramos, Daria 102 `.deepinfra` todavía es la versión rota con plan B).
- `uso.log` tiene todas las corridas con tiempos, hablantes y `costo_usd`.
