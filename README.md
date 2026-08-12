# Whisptest

Este proyecto permite la transcripción y procesamiento de audio/subtítulos con Whisper y FastAPI.

## 🚀 Instalación rápida (Entorno Virtual)

Para clonar el proyecto e instalar todas las dependencias en un solo comando:

```bash
git clone git@github.com:fdrdm/whisptest.git
cd whisptest
./setup.sh
```

El script `./setup.sh` creará automáticamente el entorno virtual (`.venv`) e instalará todas las librerías necesarias.

## 💡 Uso del entorno virtual

Una vez finalizada la instalación, activa el entorno virtual ejecutando:

```bash
source .venv/bin/activate
```

Para iniciar el servidor principal:

```bash
nohup python3 server_separation.py > server_separation.log 2>&1 & nohup python3 server_translate.py > server_translate.log 2>&1 &
```
