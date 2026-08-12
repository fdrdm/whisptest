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
echo "✅ ¡Entorno virtual preparado con éxito!"
echo "💡 Para activarlo manualmente ejecuta: source .venv/bin/activate"
echo "=========================================="
