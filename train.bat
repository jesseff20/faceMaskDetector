@echo off
REM Script para treinar o modelo de detecção de máscaras no Windows

echo 🧠 Face Mask Detector - Treinamento do Modelo
echo ===============================================

REM Verifica se o ambiente virtual existe
if not exist "venv\" (
    echo ❌ Ambiente virtual não encontrado!
    echo Execute primeiro: python setup.py
    pause
    exit /b 1
)

REM Ativa o ambiente virtual
echo 🔄 Ativando ambiente virtual...
call venv\Scripts\activate.bat

REM Verifica dependências
echo 🔍 Verificando dependências...
pip show tensorflow >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ TensorFlow não está instalado!
    echo Instalando dependências...
    pip install -r requirements.txt
)

REM Cria diretório do modelo se não existir
if not exist "model\" mkdir model
if not exist "data\" mkdir data

REM Inicia o treinamento
echo 🚀 Iniciando treinamento do modelo...
echo ⏰ Este processo pode demorar alguns minutos...
echo.

python train_model.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Treinamento concluído com sucesso!
    echo 📁 Modelo salvo em: model\mask_detector.h5
    echo.
    echo 🎯 Próximo passo: Execute run.bat para iniciar a aplicação
) else (
    echo.
    echo ❌ Erro durante o treinamento!
    echo Verifique as mensagens acima para mais detalhes.
)

echo.
pause