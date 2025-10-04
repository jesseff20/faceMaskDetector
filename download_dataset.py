#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para demonstrar o uso do kagglehub para carregar dataset de Face Mask Detection
Este script baixa e organiza o dataset andrewmvd/face-mask-detection do Kaggle
"""

import sys
import os

# Configura codificação UTF-8 para Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    os.environ["PYTHONIOENCODING"] = "utf-8"

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

def download_and_load_dataset():
    """
    Baixa e carrega o dataset de Face Mask Detection do Kaggle
    """
    
    print("[INFO] Iniciando download do dataset Face Mask Detection...")
    print("[INFO] Dataset: andrewmvd/face-mask-detection")
    print("=" * 50)
    
    try:
        # Download do dataset completo
        print("[INFO] Baixando dataset completo...")
        dataset_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
        print(f"[SUCCESS] Dataset baixado em: {dataset_path}")
        
        # Lista arquivos disponíveis
        print("\n[INFO] Arquivos disponíveis no dataset:")
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                files.append(file_path)
                print(f"   [FILE] {filename}")
        
        print(f"\n[INFO] Total de arquivos: {len(files)}")
        
        # Tenta carregar arquivo de anotações se existir
        annotation_files = [f for f in files if f.endswith('.csv') or 'annotation' in f.lower()]
        
        if annotation_files:
            print(f"\n[INFO] Arquivo de anotacoes encontrado: {annotation_files[0]}")
            
            try:
                # Carrega usando pandas adapter
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    "andrewmvd/face-mask-detection",
                    os.path.basename(annotation_files[0])
                )
                
                print("[SUCCESS] Dataset carregado como DataFrame!")
                print(f"[INFO] Shape: {df.shape}")
                print("\n[INFO] Primeiras 5 linhas:")
                print(df.head())
                
                if 'class' in df.columns or 'label' in df.columns:
                    class_col = 'class' if 'class' in df.columns else 'label'
                    print(f"\n[INFO] Distribuicao de classes ({class_col}):")
                    print(df[class_col].value_counts())
                
                return df, dataset_path
                
            except Exception as e:
                print(f"[WARNING] Erro ao carregar como DataFrame: {e}")
                print("[INFO] Dataset baixado mas nao foi possivel carregar como DataFrame")
                return None, dataset_path
        else:
            print("[INFO] Nenhum arquivo CSV de anotacoes encontrado")
            print("[INFO] Dataset contem apenas imagens")
            return None, dataset_path
            
    except Exception as e:
        print(f"[ERROR] Erro no download: {e}")
        print("\n[INFO] Possiveis solucoes:")
        print("1. Verifique sua conexao com a internet")
        print("2. Configure suas credenciais do Kaggle:")
        print("   - Baixe kaggle.json do seu perfil Kaggle")
        print("   - Coloque em ~/.kaggle/kaggle.json (Linux/Mac) ou C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)")
        print("3. Instale kagglehub: pip install kagglehub")
        return None, None

def organize_dataset_locally(dataset_path, local_data_dir="data"):
    """
    Organiza o dataset baixado na pasta local 'data'
    """
    
    if not dataset_path or not os.path.exists(dataset_path):
        print("[ERROR] Caminho do dataset invalido")
        return False
    
    print(f"\n[INFO] Organizando dataset na pasta '{local_data_dir}'...")
    
    try:
        import shutil
        
        # Remove pasta data existente
        if os.path.exists(local_data_dir):
            shutil.rmtree(local_data_dir)
            print(f"[INFO] Pasta '{local_data_dir}' existente removida")
        
        # Copia dataset para pasta local
        shutil.copytree(dataset_path, local_data_dir)
        print(f"[SUCCESS] Dataset copiado para: {os.path.abspath(local_data_dir)}")
        
        # Lista estrutura da pasta
        print(f"\n[INFO] Estrutura da pasta '{local_data_dir}':")
        for root, dirs, files in os.walk(local_data_dir):
            level = root.replace(local_data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}[FOLDER] {os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Mostra apenas primeiros 5 arquivos
                print(f"{subindent}[FILE] {file}")
            if len(files) > 5:
                print(f"{subindent}... e mais {len(files) - 5} arquivos")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Erro ao organizar dataset: {e}")
        return False

def main():
    """
    Função principal para download e organização do dataset
    """
    
    print("[INFO] KAGGLE DATASET DOWNLOADER - FACE MASK DETECTION")
    print("=" * 60)
    
    # Download e carregamento
    df, dataset_path = download_and_load_dataset()
    
    if dataset_path:
        # Organiza localmente
        if organize_dataset_locally(dataset_path):
            print("\n[SUCCESS] Dataset organizado com sucesso!")
            print("\n[INFO] Proximos passos:")
            print("1. Use as imagens da pasta 'data' para treinar seu modelo")
            print("2. Execute 'python train_model.py' para treinar o modelo CNN")
            print("3. Execute 'streamlit run app.py' para testar a aplicacao")
        else:
            print(f"[WARNING] Dataset baixado em: {dataset_path}")
            print("[INFO] Voce pode usar os arquivos diretamente desta localizacao")
    else:
        print("\n[ERROR] Download falhou. Tente novamente ou baixe manualmente:")
        print("[INFO] URL: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")

if __name__ == "__main__":
    main()