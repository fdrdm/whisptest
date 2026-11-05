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

## 💡 Ejecutar los servidores

Activa el entorno virtual:

```bash
source .venv/bin/activate
```

Inicia los servidores:

```bash
nohup python3 server_separation.py > server_separation.log 2>&1 &
nohup python3 server_translate.py > server_translate.log 2>&1 &
```

## 🌐 Accesos Web

Una vez iniciados los servidores, puedes acceder a cada interfaz en el navegador:

- **Server Separation**: [http://localhost:3838](http://localhost:3838)
- **Server Translate**: [http://localhost:8188](http://localhost:8188)
