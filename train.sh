#!/bin/bash
# Script para treinar o modelo de detecção de máscaras no Linux/Mac

echo "🧠 Face Mask Detector - Treinamento do Modelo"
echo "==============================================="

# Verifica se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado!"
    echo "Execute primeiro: python setup.py"
    exit 1
fi

# Ativa o ambiente virtual
echo "🔄 Ativando ambiente virtual..."
source venv/bin/activate

# Verifica dependências
echo "🔍 Verificando dependências..."
if ! pip show tensorflow > /dev/null 2>&1; then
    echo "❌ TensorFlow não está instalado!"
    echo "Instalando dependências..."
    pip install -r requirements.txt
fi

# Cria diretórios se não existirem
mkdir -p model
mkdir -p data

# Inicia o treinamento
echo "🚀 Iniciando treinamento do modelo..."
echo "⏰ Este processo pode demorar alguns minutos..."
echo ""

python train_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Treinamento concluído com sucesso!"
    echo "📁 Modelo salvo em: model/mask_detector.h5"
    echo ""
    echo "🎯 Próximo passo: Execute ./run.sh para iniciar a aplicação"
else
    echo ""
    echo "❌ Erro durante o treinamento!"
    echo "Verifique as mensagens acima para mais detalhes."
fi

echo ""