# Guia do KaggleHub - Face Mask Detector

## Visão     "annotations.csv"  # arquivo específicoeral

O **KaggleHub** é uma biblioteca Python que facilita o download e carregamento de datasets do Kaggle diretamente no seu código. Este projeto utiliza o dataset `andrewmvd/face-mask-detection` para treinar o modelo de detecção de máscaras.

## Configuração Inicial

### 1. Credenciais do Kaggle

Para usar o KaggleHub, você precisa configurar suas credenciais do Kaggle:

1. **Criar conta**: [kaggle.com](https://www.kaggle.com)
2. **Gerar API Token**: Account  Create New API Token
3. **Baixar kaggle.json**: Salve o arquivo de credenciais
4. **Posicionar arquivo**:
   - **Windows**: `C:\Users\<seu-user>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

### 2. Instalação

```bash
pip install kagglehub
```

## Métodos de Download

### Método 1: Download Completo do Dataset

```python
import kagglehub

# Download de todo o dataset
path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
print(f"Dataset baixado em: {path}")

# Lista arquivos baixados
import os
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
```

### Método 2: Carregamento com Pandas Adapter

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Carrega arquivo específico como DataFrame
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "andrewmvd/face-mask-detection",
    "annotations.csv"  # arquivo específica
)

print("Primeiras 5 linhas:")
print(df.head())

print("Informações do dataset:")
print(df.info())
```

### Método 3: Script Automatizado (Recomendado)

```bash
# Use o script fornecido
python download_dataset.py

# Ou execute via batch/shell
# Windows: download_data.bat
# Linux/Mac: ./download_data.sh
```

##  Estrutura do Dataset

O dataset `andrewmvd/face-mask-detection` contém:

```
face-mask-detection/
  images/              # Imagens originais
    maksssksksss0.png
    maksssksksss1.png
    ...
  annotations.csv      # Anotações (opcional)
  train.csv           # Labels de treino (opcional)
  test.csv            # Labels de teste (opcional)
```

## Integração com o Projeto

### 1. Organização Automática

O script `download_dataset.py` organiza automaticamente os dados:

```python
# Dataset original → pasta local 'data'
path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
shutil.copytree(path, "data")
```

### 2. Carregamento no Treinamento

O `train_model.py` verifica automaticamente:

1. **Dataset local** (`data/` directory)
2. **Download via KaggleHub** (se necessário)
3. **Dataset sintético** (fallback)

```python
# Verificação automática
has_local, local_path = self.check_local_dataset()
if has_local:
    # Usa dataset local
    dataset_path = local_path
else:
    # Baixa do Kaggle
    dataset_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
```

##  Casos de Uso

### Caso 1: Primeira Execução

```bash
# Configuração completa com download
python setup.py
#  Responder "s" para download do dataset
#  Responder "s" para treinamento
```

### Caso 2: Download Posterior

```bash
# Se no baixou durante setup
python download_dataset.py

# Depois treinar
python train_model.py
```

### Caso 3: Desenvolvimento/Debug

```python
# Carregamento rápido para análise
import kagglehub

# S lista arquivos sem download completo
metadata = kagglehub.dataset_metadata("andrewmvd/face-mask-detection")
print(metadata)
```

## Solução de Problemas

### Erro: "Dataset not found"
```bash
# Verifica se dataset existe
kagglehub search "face-mask-detection"
```

### Erro: "Authentication failed"
```bash
# Verifica credenciais
cat ~/.kaggle/kaggle.json  # Linux/Mac
type %USERPROFILE%\.kaggle\kaggle.json  # Windows
```

### Erro: "Permission denied"
```bash
# Linux/Mac: ajusta permissões
chmod 600 ~/.kaggle/kaggle.json
```

### Download lento
```python
# Download com progresso
import kagglehub
kagglehub.dataset_download(
    "andrewmvd/face-mask-detection",
    quiet=False  # Mostra progresso
)
```


## Informações do Dataset

- **Nome**: Face Mask Detection
- **Autor**: Andrew MV
- **Tamanho**: ~100MB
- **Imagens**: ~853 imagens
- **Classes**: com_máscara, sem_máscara
- **Formato**: PNG, JPG
- **Resolução**: Variada

## Links Úteis

- **Dataset**: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **KaggleHub Docs**: https://github.com/Kaggle/kagglehub
- **Kaggle API**: https://www.kaggle.com/docs/api

##  Dicas de Performance

1. **Cache Local**: O KaggleHub faz cache automático
2. **Download Único**: Evite downloads repetidos
3. **Organização**: Use a pasta `data/` para consistência
4. **Verificação**: Sempre verifique se dados existem localmente primeiro

---

**Desenvolvido por**: Jesse Fernandes (jesseff20@gmail.com)  
**Data**: Outubro 2025
