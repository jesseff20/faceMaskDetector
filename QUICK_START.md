# 🚀 Face Mask Detector - Guia Rápido

## ⚡ Início Rápido

### 1️⃣ Primeira Execução
```bash
# Execute apenas uma vez para configurar tudo
python setup.py

# ✅ Durante a configuração:
# 📥 Responda "s" para baixar dataset real do Kaggle (recomendado)
# 🧠 Responda "s" para treinar modelo com dados reais
```

**⚙️ Configuração do Kaggle (necessária para dataset real):**
1. Crie conta em [kaggle.com](https://www.kaggle.com)
2. Baixe `kaggle.json` (Account → API → Create New Token)
3. Coloque em:
   - **Windows**: `C:\Users\<user>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

### 2️⃣ Executar a Aplicação
```bash
# Windows
run.bat

# Ou manualmente:
.\venv\Scripts\activate
streamlit run app.py
```

### 3️⃣ Acessar no Navegador
- **Local**: http://localhost:8501
- **Rede**: http://192.168.0.36:8501

---

## 🎯 Como Usar

### 📸 Upload de Imagem
1. Clique em "Browse files"
2. Selecione uma foto
3. Veja o resultado automaticamente

### 🎥 Webcam (Foto)
1. Escolha a câmera na barra lateral
2. Clique em "Take a picture"
3. Autorize acesso à câmera
4. Análise automática

### 📹 Tempo Real
1. Selecione a câmera
2. "Iniciar Câmera" → análise contínua
3. "Parar Câmera" quando terminar

---

## 🔧 Configurações

### Ajustar Precisão
- **Limite de confiança**: 0.3-1.0 (padrão: 0.8)
- **Menor valor**: Mais detecções, alguns falsos positivos
- **Maior valor**: Menos detecções, maior precisão
- **Novo padrão 80%**: Melhor precisão nas detecções

### Múltiplas Câmeras
- Sistema detecta automaticamente
- Escolha na barra lateral
- Botão "Testar Câmera" para verificar

---

## 🎨 Interpretação dos Resultados

### Cores das Caixas
- 🟢 **Verde**: Pessoa COM máscara
- 🔴 **Vermelho**: Pessoa SEM máscara

### 🎉 Feedback Positivo
- **Todos com máscara**: Mensagem especial de parabenização
- **Animação verde**: Efeito visual positivo
- **Reconhecimento**: Agradecimento por contribuir com a segurança

### Métricas
- **👥 Rostos Detectados**: Total de pessoas
- **😷 Com Máscara**: Pessoas usando máscara
- **😐 Sem Máscara**: Pessoas sem máscara

---

## 🚨 Solução de Problemas

### Câmera não funciona
- Verifique permissões no navegador
- Teste com botão "🔍 Testar Câmera"
- Feche outros apps usando câmera

### Aplicação lenta
- Aumente limite de confiança
- Feche outros programas
- Use resolução menor

### Modelo impreciso
```bash
# Treine modelo melhor
python train_model.py
```

---

## 📊 Status do Sistema

✅ **Funcionando**: Sistema carregado e pronto  
✅ **Modelo**: Demonstração carregado  
✅ **Câmeras**: Detectadas automaticamente  
✅ **Interface**: Streamlit rodando  

---

## 🔄 Atualizações

### Melhorar Modelo
```bash
python train_model.py    # Treina com dados reais
```

### Atualizar Dependências
```bash
pip install -r requirements.txt --upgrade
```

---

## 🎭 Enjoy!

**Sistema pronto para uso!** 🚀  
Acesse: **http://localhost:8501**

---

## 👨‍💻 Desenvolvedor

**Jesse Fernandes**  
📧 jesseff20@gmail.com

*Sistema de Detecção de Máscaras Faciais com Deep Learning*