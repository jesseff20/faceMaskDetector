# 🎭 Face Mask Detector

Um sistema inteligente de detecção de máscaras faciais usando **Deep Learning** e **Computer Vision**. O projeto utiliza redes neurais convolucionais (CNN) para identificar se uma pessoa está usando máscara facial, não usando ou usando incorretamente em tempo real através de webcam ou análise de imagens.

## � Objetivo Principal

Este projeto é focado no **treinamento de modelos de Deep Learning** para detecção de máscaras faciais. O volume de imagens e a estratégia de balanceamento são fatores determinantes para alcançar **alta acurácia** e **baixo loss** no modelo final.

### 📊 Relação Volume vs Performance
- **Volume de Imagens**: Diretamente proporcional à acurácia do modelo
- **Balanceamento de Classes**: Essencial para evitar overfitting
- **Meta de Acurácia**: Sistema permite definir acurácia alvo (70-99%)
- **Loss Otimizado**: Técnicas avançadas para minimizar o erro

## �🌟 Características

- 🔍 **Detecção em Tempo Real**: Análise via webcam com processamento ao vivo
- 📸 **Upload de Imagens**: Análise de fotos carregadas pelo usuário
- 📷 **Captura de Fotos**: Tire fotos diretamente pela webcam
- 🎯 **Treinamento Inteligente**: Sistema adaptativo com meta de acurácia
- 🏗️ **3 Classes de Detecção**: Com máscara, sem máscara, máscara incorreta
- 🌐 **Interface Web**: Interface amigável desenvolvida em Streamlit
- ⚡ **Processamento Otimizado**: Múltiplos detectores de rostos para maior precisão
- 📊 **Métricas Avançadas**: Estatísticas detalhadas e análise de performance

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| **Python** | 3.7+ | Linguagem principal |
| **Streamlit** | 1.28.0 | Interface web |
| **TensorFlow** | 2.13.0 | Framework de deep learning |
| **Keras** | 2.13.1 | API de alto nível para redes neurais |
| **OpenCV** | 4.8.1 | Processamento de imagens e visão computacional |
| **NumPy** | 1.24.3 | Operações numéricas |
| **Matplotlib** | 3.7.2 | Visualização de dados |
| **Scikit-learn** | 1.3.0 | Métricas de avaliação |

## 📁 Estrutura do Projeto

```
faceMaskDetector/
├── 📁 model/                    # Modelos treinados e arquivos relacionados
│   ├── mask_detector.h5         # Modelo principal (gerado após treinamento)
│   ├── model_info.json          # Informações do modelo
│   ├── training_history.png     # Gráficos do treinamento
│   ├── deploy.prototxt          # Configuração do detector de rostos
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Pesos do detector
├── 📁 data/                     # Dados de treinamento
│   ├── with_mask/               # Imagens com máscaras
│   └── without_mask/            # Imagens sem máscaras
├── 📄 app.py                    # Interface principal Streamlit
├── 📄 utils.py                  # Funções utilitárias
├── 📄 train_model.py            # Script de treinamento do modelo
├── 📄 setup.py                  # Configuração automática do ambiente
├── 📄 requirements.txt          # Dependências do projeto
├── 📄 README.md                 # Este arquivo
└── 📄 .gitignore               # Arquivos ignorados pelo Git
```

## 🚀 Instalação e Configuração

### Método 1: Configuração Automática (Recomendado)

1. **Clone o repositório:**
```bash
git clone <https://github.com/jesseff20/faceMaskDetector>
cd faceMaskDetector
```

2. **Execute o script de configuração:**
```bash
python setup.py
```

Este script irá:
- ✅ Verificar a versão do Python
- ✅ Criar ambiente virtual
- ✅ Instalar todas as dependências (incluindo kagglehub)
- ✅ Oferecer download do dataset real do Kaggle
- ✅ Criar modelo de demonstração funcional
- ✅ Criar arquivos básicos do projeto

### 📥 Download do Dataset de Imagens

Para treinar o modelo com dados reais, você pode baixar o dataset do Kaggle:

#### Opção A: Durante a configuração inicial
```bash
python setup.py
# Responda "s" quando perguntado sobre download do dataset
```

#### Opção B: Download posterior
```bash
# Windows
download_data.bat

# Linux/Mac  
./download_data.sh

# Ou diretamente com Python
python download_dataset.py
```

#### Opção C: Kagglehub direto
```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Download do dataset completo
path = kagglehub.dataset_download("andrewmvd/face-mask-detection") 

# Ou carregamento específico de arquivo
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "andrewmvd/face-mask-detection",
    "annotations.csv"  # arquivo específico
)
```

**⚙️ Configuração do Kaggle (necessária):**
1. Crie conta no [Kaggle](https://www.kaggle.com)
2. Vá em Account → Create New API Token
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

### Método 2: Configuração Manual

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

3. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

## 🎯 Como Usar

### 1. Treinar o Modelo (Essencial)

**🚀 O treinamento é o coração do projeto!** Execute para criar seu modelo personalizado:

```bash
python train_model.py
```

## 🎯 Sistema Inteligente de Treinamento

### **📊 Estatísticas do Dataset (Exemplo)**
```
[INFO] ESTATISTICAS DO DATASET:
   - Total de imagens: 853
   - Com mascara: 698
   - Sem mascara: 119  
   - Mascara incorreta: 36
```

### **🎮 Controle Interativo**

**1. Escolha do Volume de Imagens:**
```
[PERGUNTA] Quantas imagens carregar? (Enter para 400): 853

[PERGUNTA] Estrategia de carregamento:
1. Balanceado (max 108 imagens)
2. Todas as disponiveis (853 imagens desbalanceadas)  
Escolha (1/2, Enter para 1): 2
```

**2. Meta de Acurácia:**
```
[PERGUNTA] Qual acuracia deseja atingir? (0-100%, Enter para 80%): 90
[INFO] Treinamento com meta de acuracia: 90.0%
```

### **⚖️ Impacto do Volume de Imagens**

| Volume | Balanceamento | Acurácia Esperada | Loss Esperado | Recomendação |
|--------|---------------|-------------------|---------------|--------------|
| 108 imagens | ✅ Perfeito (100%) | 70-85% | Alto (1.0-2.0) | Ideal para testes |
| 400 imagens | ⚠️ Médio (40%) | 75-90% | Médio (0.5-1.0) | Bom compromisso |
| 853 imagens | ❌ Baixo (5%) | 80-95% | Baixo (0.1-0.5) | Máxima performance |

### **🔧 Técnicas Avançadas Implementadas**

O script de treinamento inclui:
- 📥 **Download Automático**: Dataset real do Kaggle (853 imagens)
- �️ **Class Weights**: Balanceamento automático de classes desbalanceadas  
- 🔄 **Data Augmentation**: Rotação, zoom, brilho para aumentar dataset
- � **Learning Rate Scheduler**: Ajuste dinâmico da taxa de aprendizado
- 🎯 **Early Stopping**: Para quando atinge a meta de acurácia
- 💾 **Checkpoint**: Salva melhor modelo automaticamente

### 2. Executar a Aplicação

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

### 3. Usar a Interface

**📸 Upload de Imagem:**
- Clique em "Browse files" 
- Selecione uma imagem (PNG, JPG, JPEG)
- Veja o resultado da análise

**🎥 Webcam (Foto):**
- Clique em "Take a picture"
- Autorize o acesso à câmera
- Tire uma foto e veja a análise

**📹 Webcam (Tempo Real):**
- Clique em "Iniciar Câmera"
- Veja a análise em tempo real
- Clique em "Parar Câmera" para finalizar

## 🧠 Arquitetura do Modelo

O modelo utiliza uma **CNN (Convolutional Neural Network)** avançada com **3 classes de saída**:

```
📊 Arquitetura da CNN Atualizada:

Input Layer (224x224x3)
        ↓
Conv2D (32 filters, 3x3) + BatchNorm + ReLU
        ↓
MaxPooling2D (2x2) + Dropout (0.25)
        ↓
Conv2D (64 filters, 3x3) + BatchNorm + ReLU
        ↓
MaxPooling2D (2x2) + Dropout (0.25)
        ↓
Conv2D (128 filters, 3x3) + BatchNorm + ReLU
        ↓
MaxPooling2D (2x2) + Dropout (0.25)
        ↓
Conv2D (128 filters, 3x3) + BatchNorm + ReLU
        ↓
MaxPooling2D (2x2) + Dropout (0.25)
        ↓
Flatten
        ↓
Dense (512 neurons) + BatchNorm + ReLU + Dropout (0.5)
        ↓
Dense (3 neurons) + Softmax
        ↓
Output: [Com Máscara, Sem Máscara, Máscara Incorreta]
```

### **🎯 Especificações Técnicas**

| Especificação | Valor | Descrição |
|---------------|-------|-----------|
| **Input Size** | 224x224x3 | Imagens RGB redimensionadas |
| **Classes** | 3 classes | Com máscara, sem máscara, máscara incorreta |
| **Parâmetros** | ~9.6M | Total de parâmetros treináveis |
| **Arquitetura** | 4 blocos CNN | Extração hierárquica de features |
| **Otimizador** | Adam | Learning rate adaptativo |
| **Loss Function** | Categorical Crossentropy | Para classificação multiclasse |

### **� Otimizações de Performance**

**📈 Data Augmentation Intensivo:**
- Rotação: ±30°
- Zoom: ±20%
- Translação: ±30%
- Brilho: 80-120%
- Espelhamento horizontal

**⚖️ Balanceamento de Classes:**
- Class weights automáticos
- Sampling inteligente
- Estratégia balanceada vs desbalanceada

**📚 Learning Rate Strategy:**
- Scheduler adaptativo
- Redução automática no plateau
- Early stopping com meta de acurácia

## 📊 Funcionalidades da Interface

### Painel Lateral
- ⚙️ **Configurações**: Ajustes de confiança e visualização
- 📋 **Informações do Modelo**: Status e métricas do modelo carregado
- 🎛️ **Controles Avançados**: Limiar de confiança, opções de exibição

### Área Principal
- 🖼️ **Visualização**: Imagens originais e processadas lado a lado
- 📊 **Métricas**: Contadores de rostos detectados
- 🔍 **Detalhes**: Informações detalhadas de cada detecção
- 🎨 **Bounding Boxes**: Caixas coloridas ao redor dos rostos
  - 🟢 **Verde**: Com máscara
  - 🔴 **Vermelho**: Sem máscara

## 🔧 Configurações Avançadas

### Ajuste de Confiança
- **Padrão**: 0.5 (50%)
- **Uso**: Filtra detecções com baixa confiança
- **Recomendação**: Valores entre 0.3-0.7 para melhor performance

### Otimização de Performance
```python
# Para melhor performance em tempo real
confidence_threshold = 0.6  # Reduz falsos positivos
fps_limit = 30              # Limita FPS para economizar CPU
```

## 📈 Dataset e Performance do Treinamento

### 📊 Impacto Crítico do Volume de Dados

**O volume de imagens é o fator determinante para acurácia e baixo loss!**

#### Dataset Real - Kaggle Face Mask Detection
- **Fonte**: [Kaggle - Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)
- **Total**: 853 imagens com anotações XML
- **Distribuição Real**:
  - ✅ Com máscara: 698 imagens (81.8%)
  - ❌ Sem máscara: 119 imagens (13.9%) 
  - ⚠️ Máscara incorreta: 36 imagens (4.3%)

### 🎯 Cenários de Performance vs Volume

| Estratégia | Volume | Balanceamento | Acurácia Final | Loss Final | Tempo |
|-----------|--------|---------------|----------------|------------|-------|
| **Teste** | 108 imgs | 🟢 Perfeito (100%) | 70-85% | 0.8-1.5 | 10min |
| **Dev** | 400 imgs | 🟡 Médio (40%) | 75-90% | 0.4-0.8 | 25min |
| **Produção** | 853 imgs | 🔴 Baixo (5%) | 85-95% | 0.1-0.4 | 60min |

### 📊 Métricas Observadas no Sistema

#### ✅ Modelo Balanceado (108 imagens)
```
[SUCCESS] Dataset Kaggle carregado:
   - Total de imagens: 108
   - Com mascara: 36
   - Sem mascara: 36
   - Mascara incorreta: 36
   - Balanceamento: 100.00%
```
- **Acurácia**: 70-85% (estável)
- **Loss**: 0.8-1.5 (controlado)
- **Convergência**: Rápida (10-20 épocas)

#### 🚀 Modelo Desbalanceado (853 imagens)  
```
[SUCCESS] Dataset Kaggle carregado:
   - Total de imagens: 853
   - Com mascara: 698
   - Sem mascara: 119
   - Mascara incorreta: 36
   - Balanceamento: 5.16%
```
- **Acurácia**: 85-95% (máxima)
- **Loss**: 0.1-0.4 (muito baixo)
- **Convergência**: Lenta (50-100 épocas)
- ⚖️ **F1-Score**: ~94-97%

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

## 🚨 Solução de Problemas

### Problemas Comuns

**❌ Erro: "No module named 'cv2'"**
```bash
pip install opencv-python
```

**❌ Erro: "Could not find a version that satisfies TensorFlow"**
- Verifique se está usando Python 3.7-3.11
- Use: `pip install tensorflow==2.13.0`

**❌ Câmera não funciona**
- Verifique permissões da câmera no navegador
- Teste com `cv2.VideoCapture(0)` em outro script

**❌ Modelo não carregado**
- Execute `python train_model.py` primeiro
- Verifique se `model/mask_detector.h5` existe

### ⚡ Otimização de Performance

**🐌 Treinamento lento**
- Use GPU se disponível: `pip install tensorflow-gpu`
- Reduza volume de imagens para testes
- Use estratégia balanceada para convergência rápida

**💾 Alto uso de memória durante treino**
- Reduza batch_size no código
- Feche outras aplicações
- Use imagens menores (redimensionar dataset)

**📊 Baixa acurácia do modelo**
- **Volume insuficiente**: Use mais imagens (recomendado >400)
- **Desbalanceamento**: Escolha estratégia balanceada  
- **Meta muito alta**: Reduza acurácia alvo (70-85% realista)
- **Overfitting**: Adicione mais dropout ou regularização

**🎯 Como melhorar Loss**
- **Mais dados**: Volume maior = loss menor
- **Class weights**: Já implementado automaticamente
- **Data augmentation**: Já otimizado no sistema
- **Learning rate**: Sistema adaptativo implementado

## 🛡️ Considerações de Segurança

- 🔒 **Privacidade**: Imagens não são armazenadas permanentemente
- 🌐 **Local**: Todo processamento é feito localmente
- 📱 **Câmera**: Acesso à câmera apenas quando autorizado
- 🔐 **Dados**: Nenhum dado é enviado para serviços externos

## 🎨 Personalização

### Modificar Cores
```python
# Em app.py, seção CSS
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
    'epochs': 20,           # Número de épocas
    'learning_rate': 1e-4   # Taxa de aprendizado
}
```

## 🎓 Aprendizados Principais

### 💡 Conclusões do Projeto

1. **Volume é Rei**: Mais imagens = melhor modelo (acurácia ↑, loss ↓)
2. **Balanceamento vs Volume**: Trade-off crítico entre estabilidade e performance
3. **Meta Realista**: 85-90% é excelente para datasets reais desbalanceados
4. **Otimizações Funcionam**: Class weights + augmentation = grande diferença
5. **Hardware Importa**: GPU acelera significativamente o treinamento

### 🎯 Recomendações de Uso

**Para Iniciantes:**
- Use estratégia balanceada (108 imagens)
- Meta de acurácia: 75-80%
- Tempo de treino: ~10 minutos

**Para Produção:**
- Use estratégia desbalanceada (853 imagens)
- Meta de acurácia: 85-90%
- Tempo de treino: ~60 minutos

**Para Pesquisa:**
- Adicione mais datasets
- Experimente diferentes arquiteturas
- Use transfer learning

## 📚 Referências e Recursos

### 🔬 Artigos Científicos
- [Deep Learning for Face Mask Detection](https://arxiv.org/abs/2005.03950)
- [CNN Architectures for Image Classification](https://arxiv.org/abs/1409.1556)
- [Class Imbalance in Deep Learning](https://arxiv.org/abs/1901.05555)

### 📊 Datasets Alternativos
- [Medical Mask Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- [Real-World Masked Faces](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)

### 🛠️ Ferramentas Utilizadas
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

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. 🍴 Fork o projeto
2. 🌟 Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push para a branch (`git push origin feature/AmazingFeature`)
5. 🔁 Abra um Pull Request

### Áreas para Contribuição
- 🎯 Melhoria da acurácia do modelo
- 🚀 Otimização de performance
- 🎨 Melhorias na interface
- 📚 Documentação adicional
- 🧪 Testes automatizados

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## 👨‍💻 Autor

**Jesse Fernandes**  
📧 jesseff20@gmail.com

---

## 📞 Suporte

Se você encontrar problemas ou tiver dúvidas:

1. 📋 Verifique a seção [Solução de Problemas](#-solução-de-problemas)
2. 🔍 Procure em [Issues](../../issues) existentes
3. 🆕 Crie uma nova [Issue](../../issues/new) se necessário

---

**⭐ Se este projeto foi útil, considere dar uma estrela no repositório!**