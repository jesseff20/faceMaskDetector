#!/bin/bash

echo "===================================="
echo " FACE MASK DETECTOR - DATASET DOWNLOADER"
echo "===================================="
echo

echo "📥 Baixando dataset de imagens do Kaggle..."
echo "⏰ Este processo pode demorar alguns minutos..."
echo

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado!"
    echo "🔧 Execute primeiro: python setup.py"
    exit 1
fi

echo "🚀 Ativando ambiente virtual..."
source venv/bin/activate

echo "📦 Instalando kagglehub se necessário..."
pip install kagglehub --quiet

echo "🔽 Executando download do dataset..."
python download_dataset.py

if [ $? -eq 0 ]; then
    echo
    echo "✅ Download concluído com sucesso!"
    echo
    echo "📋 Próximos passos:"
    echo "1. Execute: python train_model.py"
    echo "2. Execute: streamlit run app.py"
else
    echo
    echo "❌ Erro no download"
    echo
    echo "🔧 Possíveis soluções:"
    echo "1. Configure suas credenciais do Kaggle"
    echo "2. Verifique sua conexão com a internet"
    echo "3. Baixe manualmente: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection"
fi

echo
read -p "Pressione Enter para continuar..."