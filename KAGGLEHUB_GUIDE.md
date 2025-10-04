# Guia do KaggleHub - Face Mask Detector

##  Viso Geral

O **KaggleHub**  uma biblioteca Python que facilita o download e carregamento de datasets do Kaggle diretamente no seu cdigo. Este projeto utiliza o dataset `andrewmvd/face-mask-detection` para treinar o modelo de deteco de mscaras.

##  Configurao Inicial

### 1. Credenciais do Kaggle

Para usar o KaggleHub, voc precisa configurar suas credenciais do Kaggle:

1. **Criar conta**: [kaggle.com](https://www.kaggle.com)
2. **Gerar API Token**: Account  Create New API Token
3. **Baixar kaggle.json**: Salve o arquivo de credenciais
4. **Posicionar arquivo**:
   - **Windows**: `C:\Users\<seu-user>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

### 2. Instalao

```bash
pip install kagglehub
```

##  Mtodos de Download

### Mtodo 1: Download Completo do Dataset

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

### Mtodo 2: Carregamento com Pandas Adapter

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Carrega arquivo especfico como DataFrame
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "andrewmvd/face-mask-detection",
    "annotations.csv"  # arquivo especfico
)

print("Primeiras 5 linhas:")
print(df.head())

print("Informaes do dataset:")
print(df.info())
```

### Mtodo 3: Script Automatizado (Recomendado)

```bash
# Use o script fornecido
python download_dataset.py

# Ou execute via batch/shell
# Windows: download_data.bat
# Linux/Mac: ./download_data.sh
```

##  Estrutura do Dataset

O dataset `andrewmvd/face-mask-detection` contm:

```
face-mask-detection/
  images/              # Imagens originais
    maksssksksss0.png
    maksssksksss1.png
    ...
  annotations.csv      # Anotaes (opcional)
  train.csv           # Labels de treino (opcional)
  test.csv            # Labels de teste (opcional)
```

##  Integrao com o Projeto

### 1. Organizao Automtica

O script `download_dataset.py` organiza automaticamente os dados:

```python
# Dataset original  pasta local 'data'
path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
shutil.copytree(path, "data")
```

### 2. Carregamento no Treinamento

O `train_model.py` verifica automaticamente:

1. **Dataset local** (`data/` directory)
2. **Download via KaggleHub** (se necessrio)
3. **Dataset sinttico** (fallback)

```python
# Verificao automtica
has_local, local_path = self.check_local_dataset()
if has_local:
    # Usa dataset local
    dataset_path = local_path
else:
    # Baixa do Kaggle
    dataset_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
```

##  Casos de Uso

### Caso 1: Primeira Execuo

```bash
# Configurao completa com download
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
# Carregamento rpido para anlise
import kagglehub

# S lista arquivos sem download completo
metadata = kagglehub.dataset_metadata("andrewmvd/face-mask-detection")
print(metadata)
```

##  Soluo de Problemas

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
# Linux/Mac: ajusta permisses
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


##  Informaes do Dataset

- **Nome**: Face Mask Detection
- **Autor**: Andrew MV
- **Tamanho**: ~100MB
- **Imagens**: ~853 imagens
- **Classes**: com_mscara, sem_mscara
- **Formato**: PNG, JPG
- **Resoluo**: Variada

##  Links teis

- **Dataset**: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **KaggleHub Docs**: https://github.com/Kaggle/kagglehub
- **Kaggle API**: https://www.kaggle.com/docs/api

##  Dicas de Performance

1. **Cache Local**: O KaggleHub faz cache automtico
2. **Download nico**: Evite downloads repetidos
3. **Organizao**: Use a pasta `data/` para consistncia
4. **Verificao**: Sempre verifique se dados existem localmente primeiro

---

**Desenvolvido por**: Jesse Fernandes (jesseff20@gmail.com)  
**Data**: Outubro 2025
