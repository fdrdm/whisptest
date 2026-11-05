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

Todo corre en un solo proceso (`app.py`, puerto 8188, publicado en [ia.fdrdm.com](https://ia.fdrdm.com)) con pestañas para transcribir, traducir SRT, separar audio y ver archivos e historial.

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

## 🎛️ Manejo de VRAM (`gpu.py`)

- Una sola cola de GPU: transcripción, traducción y separación nunca corren a la vez.
- Los modelos se cargan por trabajo y se descargan apenas termina (Whisper carga en ~10 s, Demucs en ~2 s); al terminar de traducir también se saca `translategemma` de Ollama.
- PyTorch queda limitado a `WHISPTEST_VRAM_GB` (5 GB por defecto, se cambia en `~/.config/systemd/user/whisptest.service`). Si un trabajo lo supera, falla con un mensaje claro en vez de dejar sin memoria al escritorio.
- Antes de empezar se verifica la VRAM libre; si no alcanza (por ejemplo, porque Ollama tiene un modelo grande cargado) se avisa qué la está usando.
- `uso.log` registra `vram_pico` de cada trabajo para calibrar `NECESARIO_GB` en `gpu.py`.
