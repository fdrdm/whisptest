#!/bin/bash
set -e

echo "=========================================="
echo "🚀 Configurando entorno virtual Python..."
echo "=========================================="

# Comprobar si uv está instalado para acelerar la instalación
if command -v uv &> /dev/null; then
    echo "⚡ 'uv' detectado. Creando entorno virtual e instalando paquetes..."
    if [ ! -d ".venv" ]; then
        uv venv .venv
    fi
    uv pip install -r requirements.txt
else
    echo "📦 Creando entorno virtual con python3 venv..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    echo "📥 Instalando dependencias con pip..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "=========================================="
echo "🦙 Verificando instalación de Ollama..."
echo "=========================================="

if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama no está instalado. Intentando instalar Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh || echo "⚠️ No se pudo instalar Ollama automáticamente. Por favor instálalo desde https://ollama.com"
fi

if command -v ollama &> /dev/null; then
    echo "📥 Descargando modelos de Ollama necesarios (translategemma:latest)..."
    ollama pull translategemma:latest || echo "⚠️ No se pudo descargar translategemma:latest. Asegúrate de que el servicio de Ollama esté ejecutándose."
else
    echo "⚠️ Recuerda instalar Ollama manualmente y descargar los modelos con:"
    echo "   ollama pull translategemma:latest"
fi

echo "=========================================="
echo "🎙️ Nemotron 3.5 Transcribe (llama.cpp + modelo GGUF)..."
echo "=========================================="

# llama-cpp-python todavía no trae el audio de Nemotron: se compila llama-server.
# Commit probado (el soporte de Nemotron Transcribe entró en master el 2026-11-03).
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_COMMIT="e14a779719652200cec7ad58dfa4820edc347182"
NEMOTRON_DIR="${NEMOTRON_DIR:-$HOME/modelos/nemotron-transcribe}"
NEMOTRON_REPO="maternion/Nemotron-3.5-Transcribe-30B-A4B-IQ-GGUF"

if [ ! -x "$LLAMA_DIR/build/bin/llama-server" ]; then
    [ -d "$LLAMA_DIR" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    git -C "$LLAMA_DIR" fetch origin && git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
    # La mitad de los núcleos: compilar CUDA con todos puede quedarse sin RAM
    cmake --build "$LLAMA_DIR/build" -j "$(( $(nproc) / 2 ))" --target llama-server
else
    echo "✔ llama-server ya está compilado en $LLAMA_DIR"
fi

# ~20 GB: el modelo IQ4_NL (los expertos van a RAM, el resto a VRAM) y el codificador de audio
.venv/bin/hf download "$NEMOTRON_REPO" \
    NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-IQ4_NL.gguf \
    mmproj-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-F16.gguf \
    --local-dir "$NEMOTRON_DIR"

echo "=========================================="
echo "✅ ¡Entorno virtual y Ollama preparados con éxito!"
echo "💡 Para activar el entorno manualmente ejecuta: source .venv/bin/activate"
echo "=========================================="
