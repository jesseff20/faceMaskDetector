@echo off
REM Script para executar o Face Mask Detector no Windows
REM Este script ativa o ambiente virtual e inicia o Streamlit

echo 🎭 Face Mask Detector - Iniciando...
echo =====================================

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

REM Verifica se o Streamlit está instalado
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Streamlit não está instalado!
    echo Instalando dependências...
    pip install -r requirements.txt
)

REM Inicia o Streamlit
echo 🚀 Iniciando aplicação Streamlit...
echo 🌐 A aplicação abrirá em: http://localhost:8501
echo.
echo ℹ️ Para parar a aplicação, pressione Ctrl+C
echo.

streamlit run app.py

echo.
echo 👋 Aplicação encerrada!
pause