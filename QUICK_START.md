# Face Mask Detector - Guia Rápido

## Início Rápido

### 1ª Primeira Execução
```bash
# Execute apenas uma vez para configurar tudo
python setup.py

# Durante a configuração:
#  Responda "s" para baixar dataset real do Kaggle (recomendado)
#  Responda "s" para treinar modelo com dados reais
```

**Configuração do Kaggle (necessária para dataset real):**
1. Crie conta em [kaggle.com](https://www.kaggle.com)
2. Baixe `kaggle.json` (Account  API  Create New Token)
3. Coloque em:
   - **Windows**: `C:\Users\<user>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

### 2. Executar a Aplicação
```bash
# Windows
run.bat

# Ou manualmente:
.\venv\Scripts\activate
streamlit run app.py
```


### 3 Acessar no Navegador
- **Local**: http://localhost:8501
- **Rede**: http://192.168.0.36:8501

---

##  Como Usar

###  Upload de Imagem
1. Clique em "Browse files"
2. Selecione uma foto
3. Veja o resultado automaticamente

###  Webcam (Foto)
1. Escolha a câmera na barra lateral
2. Clique em "Take a picture"
3. Autorize acesso à câmera
4. Análise automática

###  Tempo Real
1. Selecione a câmera
2. "Iniciar Câmera" → análise contínua
3. "Parar Câmera" quando terminar

---

##  Configuraes

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

## Interpretação dos Resultados

### Cores das Caixas
- **Verde**: Pessoa COM máscara
- **Vermelho**: Pessoa SEM máscara

### Feedback Positivo
- **Todos com máscara**: Mensagem especial de parabenização
- **Animação verde**: Efeito visual positivo
- **Reconhecimento**: Agradecimento por contribuir com a segurança

### Mtricas
- ** Rostos Detectados**: Total de pessoas
- ** Com Mscara**: Pessoas usando mscara
- ** Sem Mscara**: Pessoas sem mscara

---

##  Soluo de Problemas

### Cmera no funciona
- Verifique permisses no navegador
- Teste com boto " Testar Cmera"
- Feche outros apps usando cmera

### Aplicao lenta
- Aumente limite de confiana
- Feche outros programas
- Use resoluo menor

### Modelo impreciso
```bash
# Treine modelo melhor
python train_model.py
```

---

##  Status do Sistema

 **Funcionando**: Sistema carregado e pronto  
 **Modelo**: Demonstrao carregado  
 **Cmeras**: Detectadas automaticamente  
 **Interface**: Streamlit rodando  

---

##  Atualizaes

### Melhorar Modelo
```bash
python train_model.py    # Treina com dados reais
```

### Atualizar Dependncias
```bash
pip install -r requirements.txt --upgrade
```

---

##  Enjoy!

**Sistema pronto para uso!**   
Acesse: **http://localhost:8501**

---

##  Desenvolvedor

**Jesse Fernandes**  
 jesseff20@gmail.com

*Sistema de Deteco de Mscaras Faciais com Deep Learning*
