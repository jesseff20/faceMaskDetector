@echo off
echo ====================================
echo  FACE MASK DETECTOR - DATASET DOWNLOADER
echo ====================================
echo.

echo 📥 Baixando dataset de imagens do Kaggle...
echo ⏰ Este processo pode demorar alguns minutos...
echo.

cd /d "%~dp0"

if not exist "venv\" (
    echo ❌ Ambiente virtual não encontrado!
    echo 🔧 Execute primeiro: python setup.py
    pause
    exit /b 1
)

echo 🚀 Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo 📦 Instalando kagglehub se necessário...
pip install kagglehub --quiet

echo 🔽 Executando download do dataset...
python download_dataset.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ Download concluído com sucesso!
    echo.
    echo 📋 Próximos passos:
    echo 1. Execute: python train_model.py
    echo 2. Execute: streamlit run app.py
) else (
    echo ❌ Erro no download
    echo.
    echo 🔧 Possíveis soluções:
    echo 1. Configure suas credenciais do Kaggle
    echo 2. Verifique sua conexão com a internet
    echo 3. Baixe manualmente: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
)

echo.
pause