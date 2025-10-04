#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script simplificado para download do dataset sem problemas de codificação
"""

import sys
import os

# Força UTF-8 no Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import kagglehub
    import shutil
    
    print("Iniciando download do dataset...")
    
    # Download do dataset
    path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
    print(f"Dataset baixado em: {path}")
    
    # Move para pasta data local
    data_dir = "data"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("Pasta data existente removida")
    
    # Copia arquivos para pasta data
    shutil.copytree(path, data_dir)
    print(f"Dataset copiado para: {os.path.abspath(data_dir)}")
    
    # Lista arquivos baixados
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    print(f"Total de arquivos baixados: {len(files)}")
    print("Estrutura do dataset:")
    for file in files[:10]:  # Mostra primeiros 10 arquivos
        print(f"   {file}")
    if len(files) > 10:
        print(f"   ... e mais {len(files) - 10} arquivos")
        
    print("Download concluido com sucesso!")
    
except ImportError:
    print("ERRO: kagglehub nao esta instalado")
    print("Execute: pip install kagglehub")
    sys.exit(1)
    
except Exception as e:
    print(f"ERRO ao baixar dataset: {e}")
    print("Voce pode baixar manualmente de:")
    print("https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
    sys.exit(1)