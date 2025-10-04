#  Face Mask Detector

Um sistema inteligente de deteco de mscaras faciais usando **Deep Learning** e **Computer Vision**. O projeto utiliza redes neurais convolucionais (CNN) para identificar se uma pessoa est usando mscara facial, no usando ou usando incorretamente em tempo real atravs de webcam ou anlise de imagens.

##  Objetivo Principal

Este projeto  focado no **treinamento de modelos de Deep Learning** para deteco de mscaras faciais. O volume de imagens e a estratgia de balanceamento so fatores determinantes para alcanar **alta acurcia** e **baixo loss** no modelo final.

###  Relao Volume vs Performance
- **Volume de Imagens**: Diretamente proporcional  acurcia do modelo
- **Balanceamento de Classes**: Essencial para evitar overfitting
- **Meta de Acurcia**: Sistema permite definir acurcia alvo (70-99%)
- **Loss Otimizado**: Tcnicas avanadas para minimizar o erro

##  Caractersticas

-  **Deteco em Tempo Real**: Anlise via webcam com processamento ao vivo
-  **Upload de Imagens**: Anlise de fotos carregadas pelo usurio
-  **Captura de Fotos**: Tire fotos diretamente pela webcam
-  **Treinamento Inteligente**: Sistema adaptativo com meta de acurcia
-  **3 Classes de Deteco**: Com mscara, sem mscara, mscara incorreta
-  **Interface Web**: Interface amigvel desenvolvida em Streamlit
-  **Processamento Otimizado**: Mltiplos detectores de rostos para maior preciso
-  **Mtricas Avanadas**: Estatsticas detalhadas e anlise de performance

##  Tecnologias Utilizadas

| Tecnologia | Verso | Uso |
|------------|--------|-----|
| **Python** | 3.7+ | Linguagem principal |
| **Streamlit** | 1.28.0 | Interface web |
| **TensorFlow** | 2.13.0 | Framework de deep learning |
| **Keras** | 2.13.1 | API de alto nvel para redes neurais |
| **OpenCV** | 4.8.1 | Processamento de imagens e viso computacional |
| **NumPy** | 1.24.3 | Operaes numricas |
| **Matplotlib** | 3.7.2 | Visualizao de dados |
| **Scikit-learn** | 1.3.0 | Mtricas de avaliao |

##  Estrutura do Projeto

```
faceMaskDetector/
  model/                    # Modelos treinados e arquivos relacionados
    mask_detector.h5         # Modelo principal (gerado aps treinamento)
    model_info.json          # Informaes do modelo
    training_history.png     # Grficos do treinamento
    deploy.prototxt          # Configurao do detector de rostos
    res10_300x300_ssd_iter_140000.caffemodel  # Pesos do detector
  data/                     # Dados de treinamento
    with_mask/               # Imagens com mscaras
    without_mask/            # Imagens sem mscaras
  app.py                    # Interface principal Streamlit
  utils.py                  # Funes utilitrias
  train_model.py            # Script de treinamento do modelo
  setup.py                  # Configurao automtica do ambiente
  requirements.txt          # Dependncias do projeto
  README.md                 # Este arquivo
  .gitignore               # Arquivos ignorados pelo Git
```

##  Instalao e Configurao

### Mtodo 1: Configurao Automtica (Recomendado)

1. **Clone o repositrio:**
```bash
git clone <https://github.com/jesseff20/faceMaskDetector>
cd faceMaskDetector
```

2. **Execute o script de configurao:**
```bash
python setup.py
```

Este script ir:
-  Verificar a verso do Python
-  Criar ambiente virtual
-  Instalar todas as dependncias (incluindo kagglehub)
-  Oferecer download do dataset real do Kaggle
-  Criar modelo de demonstrao funcional
-  Criar arquivos bsicos do projeto

###  Download do Dataset de Imagens

Para treinar o modelo com dados reais, voc pode baixar o dataset do Kaggle:

#### Opo A: Durante a configurao inicial
```bash
python setup.py
# Responda "s" quando perguntado sobre download do dataset
```

#### Opo B: Download posterior
```bash
# Windows
download_data.bat

# Linux/Mac  
./download_data.sh

# Ou diretamente com Python
python download_dataset.py
```

#### Opo C: Kagglehub direto
```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Download do dataset completo
path = kagglehub.dataset_download("andrewmvd/face-mask-detection") 

# Ou carregamento especfico de arquivo
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "andrewmvd/face-mask-detection",
    "annotations.csv"  # arquivo especfico
)
```

** Configurao do Kaggle (necessria):**
1. Crie conta no [Kaggle](https://www.kaggle.com)
2. V em Account  Create New API Token
3. Baixe o arquivo `kaggle.json`
4. Coloque em:
   - **Windows**: `C:\Users\<seu-user>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

3. **Ative o ambiente virtual:**

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Mtodo 2: Configurao Manual

1. **Criar ambiente virtual:**
```bash
python -m venv venv
```

2. **Ativar ambiente virtual:**
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependncias:**
```bash
pip install -r requirements.txt
```

##  Como Usar

### 1. Treinar o Modelo (Essencial)

** O treinamento  o corao do projeto!** Execute para criar seu modelo personalizado:

```bash
python train_model.py
```

##  Sistema Inteligente de Treinamento

### ** Estatsticas do Dataset (Exemplo)**
```
[INFO] ESTATISTICAS DO DATASET:
   - Total de imagens: 853
   - Com mascara: 698
   - Sem mascara: 119  
   - Mascara incorreta: 36
```

### ** Controle Interativo**

**1. Escolha do Volume de Imagens:**
```
[PERGUNTA] Quantas imagens carregar? (Enter para 400): 853

[PERGUNTA] Estrategia de carregamento:
1. Balanceado (max 108 imagens)
2. Todas as disponiveis (853 imagens desbalanceadas)  
Escolha (1/2, Enter para 1): 2
```

**2. Meta de Acurcia:**
```
[PERGUNTA] Qual acuracia deseja atingir? (0-100%, Enter para 80%): 90
[INFO] Treinamento com meta de acuracia: 90.0%
```

### ** Impacto do Volume de Imagens**

| Volume | Balanceamento | Acurcia Esperada | Loss Esperado | Recomendao |
|--------|---------------|-------------------|---------------|--------------|
| 108 imagens |  Perfeito (100%) | 70-85% | Alto (1.0-2.0) | Ideal para testes |
| 400 imagens |  Mdio (40%) | 75-90% | Mdio (0.5-1.0) | Bom compromisso |
| 853 imagens |  Baixo (5%) | 80-95% | Baixo (0.1-0.5) | Mxima performance |

### ** Tcnicas Avanadas Implementadas**

O script de treinamento inclui:
-  **Download Automtico**: Dataset real do Kaggle (853 imagens)
-  **Class Weights**: Balanceamento automtico de classes desbalanceadas  
-  **Data Augmentation**: Rotao, zoom, brilho para aumentar dataset
-  **Learning Rate Scheduler**: Ajuste dinmico da taxa de aprendizado
-  **Early Stopping**: Para quando atinge a meta de acurcia
-  **Checkpoint**: Salva melhor modelo automaticamente

### 2. Executar a Aplicao

```bash
streamlit run app.py
```

A aplicao abrir automaticamente no navegador em `http://localhost:8501`

### 3. Usar a Interface

** Upload de Imagem:**
- Clique em "Browse files" 
- Selecione uma imagem (PNG, JPG, JPEG)
- Veja o resultado da anlise

** Webcam (Foto):**
- Clique em "Take a picture"
- Autorize o acesso  cmera
- Tire uma foto e veja a anlise

** Webcam (Tempo Real):**
- Clique em "Iniciar Cmera"
- Veja a anlise em tempo real
- Clique em "Parar Cmera" para finalizar

##  Arquitetura do Modelo

O modelo utiliza uma **CNN (Convolutional Neural Network)** avanada com **3 classes de sada**:

```
 Arquitetura da CNN Atualizada:

Input Layer (224x224x3)
        
Conv2D (32 filters, 3x3) + BatchNorm + ReLU
        
MaxPooling2D (2x2) + Dropout (0.25)
        
Conv2D (64 filters, 3x3) + BatchNorm + ReLU
        
MaxPooling2D (2x2) + Dropout (0.25)
        
Conv2D (128 filters, 3x3) + BatchNorm + ReLU
        
MaxPooling2D (2x2) + Dropout (0.25)
        
Conv2D (128 filters, 3x3) + BatchNorm + ReLU
        
MaxPooling2D (2x2) + Dropout (0.25)
        
Flatten
        
Dense (512 neurons) + BatchNorm + ReLU + Dropout (0.5)
        
Dense (3 neurons) + Softmax
        
Output: [Com Mscara, Sem Mscara, Mscara Incorreta]
```


### ** Especificaes Tcnicas**

| Especificao | Valor | Descrio |
|---------------|-------|-----------|
| **Input Size** | 224x224x3 | Imagens RGB redimensionadas |
| **Classes** | 3 classes | Com mscara, sem mscara, mscara incorreta |
| **Parmetros** | ~9.6M | Total de parmetros treinveis |
| **Arquitetura** | 4 blocos CNN | Extrao hierrquica de features |
| **Otimizador** | Adam | Learning rate adaptativo |
| **Loss Function** | Categorical Crossentropy | Para classificao multiclasse |

### ** Otimizaes de Performance**

** Data Augmentation Intensivo:**
- Rotao: 30
- Zoom: 20%
- Translao: 30%
- Brilho: 80-120%
- Espelhamento horizontal

** Balanceamento de Classes:**
- Class weights automticos
- Sampling inteligente
- Estratgia balanceada vs desbalanceada

** Learning Rate Strategy:**
- Scheduler adaptativo
- Reduo automtica no plateau
- Early stopping com meta de acurcia

##  Funcionalidades da Interface

### Painel Lateral
-  **Configuraes**: Ajustes de confiana e visualizao
-  **Informaes do Modelo**: Status e mtricas do modelo carregado
-  **Controles Avanados**: Limiar de confiana, opes de exibio

### rea Principal
-  **Visualizao**: Imagens originais e processadas lado a lado
-  **Mtricas**: Contadores de rostos detectados
-  **Detalhes**: Informaes detalhadas de cada deteco
-  **Bounding Boxes**: Caixas coloridas ao redor dos rostos
  -  **Verde**: Com mscara
  -  **Vermelho**: Sem mscara

##  Configuraes Avanadas

### Ajuste de Confiana
- **Padro**: 0.5 (50%)
- **Uso**: Filtra deteces com baixa confiana
- **Recomendao**: Valores entre 0.3-0.7 para melhor performance

### Otimizao de Performance
```python
# Para melhor performance em tempo real
confidence_threshold = 0.6  # Reduz falsos positivos
fps_limit = 30              # Limita FPS para economizar CPU
```

##  Dataset e Performance do Treinamento

###  Impacto Crtico do Volume de Dados

**O volume de imagens  o fator determinante para acurcia e baixo loss!**

#### Dataset Real - Kaggle Face Mask Detection
- **Fonte**: [Kaggle - Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)
- **Total**: 853 imagens com anotaes XML
- **Distribuio Real**:
  -  Com mscara: 698 imagens (81.8%)
  -  Sem mscara: 119 imagens (13.9%) 
  -  Mscara incorreta: 36 imagens (4.3%)

###  Cenrios de Performance vs Volume

| Estratgia | Volume | Balanceamento | Acurcia Final | Loss Final | Tempo |
|-----------|--------|---------------|----------------|------------|-------|
| **Teste** | 108 imgs |  Perfeito (100%) | 70-85% | 0.8-1.5 | 10min |
| **Dev** | 400 imgs |  Mdio (40%) | 75-90% | 0.4-0.8 | 25min |
| **Produo** | 853 imgs |  Baixo (5%) | 85-95% | 0.1-0.4 | 60min |

###  Mtricas Observadas no Sistema

####  Modelo Balanceado (108 imagens)
```
[SUCCESS] Dataset Kaggle carregado:
   - Total de imagens: 108
   - Com mascara: 36
   - Sem mascara: 36
   - Mascara incorreta: 36
   - Balanceamento: 100.00%
```
- **Acurcia**: 70-85% (estvel)
- **Loss**: 0.8-1.5 (controlado)
- **Convergncia**: Rpida (10-20 pocas)

####  Modelo Desbalanceado (853 imagens)  
```
[SUCCESS] Dataset Kaggle carregado:
   - Total de imagens: 853
   - Com mascara: 698
   - Sem mascara: 119
   - Mascara incorreta: 36
   - Balanceamento: 5.16%
```
- **Acurcia**: 85-95% (mxima)
- **Loss**: 0.1-0.4 (muito baixo)
- **Convergncia**: Lenta (50-100 pocas)
-  **F1-Score**: ~94-97%

### Data Augmentation
```python
augmentation_config = {
    'rotation_range': 20,
    'zoom_range': 0.15,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'horizontal_flip': True
}
```

##  Soluo de Problemas

### Problemas Comuns

** Erro: "No module named 'cv2'"**
```bash
pip install opencv-python
```

** Erro: "Could not find a version that satisfies TensorFlow"**
- Verifique se est usando Python 3.7-3.11
- Use: `pip install tensorflow==2.13.0`

** Cmera no funciona**
- Verifique permisses da cmera no navegador
- Teste com `cv2.VideoCapture(0)` em outro script

** Modelo no carregado**
- Execute `python train_model.py` primeiro
- Verifique se `model/mask_detector.h5` existe

###  Otimizao de Performance

** Treinamento lento**
- Use GPU se disponvel: `pip install tensorflow-gpu`
- Reduza volume de imagens para testes
- Use estratgia balanceada para convergncia rpida

** Alto uso de memria durante treino**
- Reduza batch_size no cdigo
- Feche outras aplicaes
- Use imagens menores (redimensionar dataset)

** Baixa acurcia do modelo**
- **Volume insuficiente**: Use mais imagens (recomendado >400)
- **Desbalanceamento**: Escolha estratgia balanceada  
- **Meta muito alta**: Reduza acurcia alvo (70-85% realista)
- **Overfitting**: Adicione mais dropout ou regularizao

** Como melhorar Loss**
- **Mais dados**: Volume maior = loss menor
- **Class weights**: J implementado automaticamente
- **Data augmentation**: J otimizado no sistema
- **Learning rate**: Sistema adaptativo implementado

##  Consideraes de Segurana

-  **Privacidade**: Imagens no so armazenadas permanentemente
-  **Local**: Todo processamento  feito localmente
-  **Cmera**: Acesso  cmera apenas quando autorizado
-  **Dados**: Nenhum dado  enviado para servios externos

##  Personalizao

### Modificar Cores
```python
# Em app.py, seo CSS
colors = {
    'with_mask': '#28a745',    # Verde
    'without_mask': '#dc3545', # Vermelho
    'primary': '#1f77b4'       # Azul
}
```

### Ajustar Modelo
```python
# Em train_model.py
model_config = {
    'img_size': 224,        # Tamanho da imagem
    'batch_size': 32,       # Tamanho do batch
    'epochs': 20,           # Nmero de pocas
    'learning_rate': 1e-4   # Taxa de aprendizado
}
```

##  Aprendizados Principais

###  Concluses do Projeto

1. **Volume  Rei**: Mais imagens = melhor modelo (acurcia , loss )
2. **Balanceamento vs Volume**: Trade-off crtico entre estabilidade e performance
3. **Meta Realista**: 85-90%  excelente para datasets reais desbalanceados
4. **Otimizaes Funcionam**: Class weights + augmentation = grande diferena
5. **Hardware Importa**: GPU acelera significativamente o treinamento

###  Recomendaes de Uso

**Para Iniciantes:**
- Use estratgia balanceada (108 imagens)
- Meta de acurcia: 75-80%
- Tempo de treino: ~10 minutos

**Para Produo:**
- Use estratgia desbalanceada (853 imagens)
- Meta de acurcia: 85-90%
- Tempo de treino: ~60 minutos

**Para Pesquisa:**
- Adicione mais datasets
- Experimente diferentes arquiteturas
- Use transfer learning

##  Referncias e Recursos

###  Artigos Cientficos
- [Deep Learning for Face Mask Detection](https://arxiv.org/abs/2005.03950)
- [CNN Architectures for Image Classification](https://arxiv.org/abs/1409.1556)
- [Class Imbalance in Deep Learning](https://arxiv.org/abs/1901.05555)

###  Datasets Alternativos
- [Medical Mask Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- [Real-World Masked Faces](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)

###  Ferramentas Utilizadas
- [Kaggle API](https://github.com/Kaggle/kaggle-api) - Download de datasets
- [Streamlit](https://streamlit.io/) - Interface web
- [TensorFlow](https://tensorflow.org/) - Framework de ML
- [OpenCV](https://opencv.org/) - Processamento de imagens

### Datasets Alternativos
- [Real World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)

### Ferramentas Relacionadas
- [MediaPipe Face Detection](https://mediapipe.dev/)
- [OpenCV Face Recognition](https://opencv.org/)
- [Detectron2](https://github.com/facebookresearch/detectron2)

##  Contribuio

Contribuies so bem-vindas! Para contribuir:

1.  Fork o projeto
2.  Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3.  Commit suas mudanas (`git commit -m 'Add some AmazingFeature'`)
4.  Push para a branch (`git push origin feature/AmazingFeature`)
5.  Abra um Pull Request

### reas para Contribuio
-  Melhoria da acurcia do modelo
-  Otimizao de performance
-  Melhorias na interface
-  Documentao adicional
-  Testes automatizados

##  Licena

Este projeto est licenciado sob a [MIT License](LICENSE).

##  Autor

**Jesse Fernandes**  
 jesseff20@gmail.com

---

##  Suporte

Se voc encontrar problemas ou tiver dvidas:

1.  Verifique a seo [Soluo de Problemas](#-soluo-de-problemas)
2.  Procure em [Issues](../../issues) existentes
3.  Crie uma nova [Issue](../../issues/new) se necessrio

---

** Se este projeto foi til, considere dar uma estrela no repositrio!**
