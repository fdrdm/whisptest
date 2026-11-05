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
echo "✅ ¡Entorno virtual y Ollama preparados con éxito!"
echo "💡 Para activar el entorno manualmente ejecuta: source .venv/bin/activate"
echo "=========================================="
