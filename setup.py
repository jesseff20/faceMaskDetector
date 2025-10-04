"""
Script de configuração automática do ambiente para o projeto Face Mask Detector
Este script cria um ambiente virtual, instala as dependências e prepara o projeto
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Executa um comando no shell e verifica se foi bem-sucedido"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro em {description}: {e}")
        print(f"Saída do erro: {e.stderr}")
        return False

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Versão compatível!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Versão não compatível!")
        print("Este projeto requer Python 3.7 ou superior.")
        return False

def create_virtual_environment():
    """Cria ambiente virtual"""
    if not os.path.exists("venv"):
        return run_command("python -m venv venv", "Criando ambiente virtual")
    else:
        print("✅ Ambiente virtual já existe!")
        return True

def activate_and_install():
    """Ativa ambiente virtual e instala dependências"""
    system = platform.system().lower()
    
    if system == "windows":
        activate_cmd = ".\\venv\\Scripts\\activate"
        pip_cmd = ".\\venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "./venv/bin/pip"
    
    # Atualiza pip primeiro
    if not run_command(f"{pip_cmd} install --upgrade pip", "Atualizando pip"):
        return False
    
    # Instala dependências básicas
    dependencies = [
        "streamlit",
        "opencv-python",
        "tensorflow",
        "keras",
        "numpy",
        "pillow",
        "matplotlib",
        "scikit-learn",
        "kagglehub",
        "kaggle"
    ]
    
    for dep in dependencies:
        if not run_command(f"{pip_cmd} install {dep}", f"Instalando {dep}"):
            print(f"⚠️ Falha ao instalar {dep}, continuando...")
    
    return True

def create_project_files():
    """Cria arquivos básicos do projeto se não existirem"""
    files_to_create = {
        "requirements.txt": """streamlit==1.28.0
opencv-python==4.8.1.78
tensorflow==2.13.0
keras==2.13.1
numpy==1.24.3
pillow==10.0.1
matplotlib==3.7.2
scikit-learn==1.3.0
kagglehub==0.2.5
""",
        "utils.py": """# Funções utilitárias para detecção de máscaras
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Arquivo será implementado posteriormente
""",
        "app.py": """import streamlit as st

# Interface Streamlit - será implementada posteriormente
st.title("🎭 Face Mask Detector")
st.write("Interface em desenvolvimento...")
""",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Data
data/
*.h5
*.pkl
*.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    }
    
    for filename, content in files_to_create.items():
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Arquivo {filename} criado!")
        else:
            print(f"ℹ️ Arquivo {filename} já existe!")

def download_dataset():
    """Baixa o dataset de máscaras do Kaggle usando kagglehub"""
    print("📥 Baixando dataset de imagens do Kaggle...")
    
    # Cria diretório data se não existir
    if not os.path.exists("data"):
        os.makedirs("data")
        print("✅ Diretório 'data' criado!")
    
    system = platform.system().lower()
    if system == "windows":
        python_cmd = ".\\venv\\Scripts\\python"
    else:
        python_cmd = "./venv/bin/python"
    
    # Usa script simplificado sem problemas de codificação
    try:
        # Executa download usando script simple_download.py
        success = run_command(f"{python_cmd} simple_download.py", "Baixando dataset do Kaggle")
        return success
    except Exception as e:
        print(f"Erro no download: {e}")
        return False

def train_initial_model():
    """Treina um modelo inicial básico para demonstração"""
    print("🧠 Treinando modelo inicial...")
    
    system = platform.system().lower()
    if system == "windows":
        python_cmd = ".\\venv\\Scripts\\python"
    else:
        python_cmd = "./venv/bin/python"
    
    # Executa o script de treinamento
    return run_command(f"{python_cmd} train_model.py", "Treinando modelo CNN")

def main():
    """Função principal de configuração"""
    print("🚀 Configurando projeto Face Mask Detector")
    print("=" * 50)
    
    # Verifica versão do Python
    if not check_python_version():
        return False
    
    # Cria ambiente virtual
    if not create_virtual_environment():
        return False
    
    # Ativa ambiente e instala dependências
    if not activate_and_install():
        return False
    
    # Cria arquivos básicos do projeto
    create_project_files()
    
    # Sistema já não precisa de modelo de demonstração
    # O treinamento será feito com dados reais conforme necessário
    
    # Pergunta sobre download do dataset
    print("\n📥 DOWNLOAD DO DATASET DE IMAGENS")
    print("=" * 40)
    print("📊 Para treinar o modelo com dados reais, é necessário baixar imagens do Kaggle.")
    download_data = input("🔽 Deseja baixar o dataset de imagens agora? (s/N): ").lower().strip()
    
    dataset_downloaded = False
    if download_data in ['s', 'sim', 'y', 'yes']:
        print("\n� Iniciando download do dataset...")
        print("⏰ Este processo pode demorar alguns minutos dependendo da conexão...")
        
        if download_dataset():
            print("✅ Dataset baixado com sucesso!")
            dataset_downloaded = True
        else:
            print("⚠️ Erro no download. Você pode fazer isso depois ou baixar manualmente:")
            print("   https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
    else:
        print("ℹ️ Dataset não baixado. Para baixar depois execute:")
        print("   kagglehub dataset_download andrewmvd/face-mask-detection")

    # Pergunta sobre treinamento completo
    print("\n🧠 TREINAMENTO DO MODELO")
    print("=" * 30)
    if dataset_downloaded:
        print("📚 Dataset disponível! Agora você pode treinar o modelo com dados reais.")
        train_model = input("🚀 Deseja treinar o modelo completo agora? (s/N): ").lower().strip()
    else:
        print("📚 Para treinar com dados reais, baixe o dataset primeiro.")
        train_model = input("🧠 Deseja treinar modelo básico de demonstração? (s/N): ").lower().strip()
    
    if train_model in ['s', 'sim', 'y', 'yes']:
        print("\n🚀 Iniciando treinamento do modelo...")
        print("⏰ Este processo pode demorar alguns minutos...")
        
        if train_initial_model():
            print("✅ Modelo treinado com sucesso!")
        else:
            print("⚠️ Erro no treinamento. Você pode fazer isso depois executando:")
            print("   python train_model.py")
    else:
        print("ℹ️ Usando modelo de demonstração. Para treinar modelo depois execute:")
        print("   python train_model.py")
    
    print("\n" + "=" * 50)
    print("🎉 Configuração concluída com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. Ative o ambiente virtual:")
    
    system = platform.system().lower()
    if system == "windows":
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("2. Execute o Streamlit:")
    print("   streamlit run app.py")
    print("\n🔗 O aplicativo abrirá automaticamente no navegador!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Configuração falhou. Verifique os erros acima.")
        sys.exit(1)