#!/bin/bash
# Script para executar o Face Mask Detector no Linux/Mac
# Este script ativa o ambiente virtual e inicia o Streamlit

echo "🎭 Face Mask Detector - Iniciando..."
echo "====================================="

# Verifica se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado!"
    echo "Execute primeiro: python setup.py"
    exit 1
fi

# Ativa o ambiente virtual
echo "🔄 Ativando ambiente virtual..."
source venv/bin/activate

# Verifica se o Streamlit está instalado
if ! pip show streamlit > /dev/null 2>&1; then
    echo "❌ Streamlit não está instalado!"
    echo "Instalando dependências..."
    pip install -r requirements.txt
fi

# Inicia o Streamlit
echo "🚀 Iniciando aplicação Streamlit..."
echo "🌐 A aplicação abrirá em: http://localhost:8501"
echo ""
echo "ℹ️ Para parar a aplicação, pressione Ctrl+C"
echo ""

streamlit run app.py

echo ""
echo "👋 Aplicação encerrada!"