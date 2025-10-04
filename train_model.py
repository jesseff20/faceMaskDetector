#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para criar e treinar modelo CNN de detecção de máscaras faciais
Este script baixa o dataset, prepara os dados e treina o modelo
"""

import sys
import os

# Configura codificação UTF-8 para Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import cv2
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

class MaskModelTrainer:
    def __init__(self, img_size=224, batch_size=32):
        """
        Inicializa o treinador do modelo
        
        Args:
            img_size (int): Tamanho das imagens (img_size x img_size)
            batch_size (int): Tamanho do batch para treinamento
        """
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.data = []
        self.labels = []
        self.model = None
        
    def download_dataset(self):
        """Baixa o dataset do Kaggle"""
        print("[INFO] Baixando dataset de mascaras faciais...")
        try:
            # Download do dataset
            path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
            print(f"[SUCCESS] Dataset baixado em: {path}")
            return path
        except Exception as e:
            print(f"[ERROR] Erro ao baixar dataset: {e}")
            print("[INFO] Tentando usar dataset local...")
            return None
    
    def create_synthetic_dataset(self):
        """
        Cria um dataset sintético para demonstração se o download falhar
        """
        print("🎨 Criando dataset sintético para demonstração...")
        
        # Cria diretórios
        os.makedirs("data/with_mask", exist_ok=True)
        os.makedirs("data/without_mask", exist_ok=True)
        
        # Gera imagens sintéticas
        np.random.seed(42)
        
        for i in range(100):  # 100 imagens com máscara
            # Simula uma face com máscara (mais pixels na região inferior)
            img = np.random.randint(0, 255, (self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
            # Adiciona uma "máscara" na parte inferior
            img[int(self.IMG_SIZE*0.6):int(self.IMG_SIZE*0.9), 
                int(self.IMG_SIZE*0.2):int(self.IMG_SIZE*0.8)] = [100, 100, 200]  # Azul para máscara
            
            cv2.imwrite(f"data/with_mask/synthetic_{i}.jpg", img)
            
        for i in range(100):  # 100 imagens sem máscara
            # Simula uma face sem máscara
            img = np.random.randint(0, 255, (self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
            # Adiciona tom de pele na região da boca
            img[int(self.IMG_SIZE*0.6):int(self.IMG_SIZE*0.9), 
                int(self.IMG_SIZE*0.2):int(self.IMG_SIZE*0.8)] = [200, 180, 160]  # Tom de pele
            
            cv2.imwrite(f"data/without_mask/synthetic_{i}.jpg", img)
        
        print("[SUCCESS] Dataset sintetico criado!")
        return "data"
    
    def parse_xml_annotation(self, xml_file):
        """Extrai informações de uma anotação XML para 3 classes"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text.lower()
                
                # Mapeia nomes para 3 classes
                if 'with_mask' in name or name == 'mask':
                    label = 'with_mask'
                elif 'without_mask' in name or 'no_mask' in name:
                    label = 'without_mask'
                elif 'mask_weared_incorrect' in name or 'incorrect' in name or 'wrong' in name:
                    label = 'mask_weared_incorrect'
                else:
                    # Tenta inferir pelo contexto - padrão: without_mask
                    if 'mask' in name:
                        label = 'with_mask'
                    else:
                        label = 'without_mask'
                
                # Extrai bounding box se necessário (para futuras melhorias)
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    objects.append({
                        'label': label,
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
                else:
                    objects.append({'label': label, 'bbox': None})
            
            return objects
        except Exception as e:
            print(f"[ERROR] Erro ao processar XML {xml_file}: {e}")
            return []
    
    def count_dataset_images(self, data_dir, dataset_type):
        """Conta o total de imagens disponíveis no dataset para 3 classes"""
        total_images = 0
        with_mask_count = 0
        without_mask_count = 0
        mask_incorrect_count = 0
        
        if dataset_type == "kaggle":
            images_dir = os.path.join(data_dir, "images")
            annotations_dir = os.path.join(data_dir, "annotations")
            
            if os.path.exists(images_dir):
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
                
                for image_file in image_files:
                    base_name = os.path.splitext(image_file)[0]
                    xml_file = os.path.join(annotations_dir, base_name + '.xml')
                    
                    if os.path.exists(xml_file):
                        annotations = self.parse_xml_annotation(xml_file)
                        if annotations:
                            label = annotations[0]['label']
                            if label == 'with_mask':
                                with_mask_count += 1
                            elif label == 'mask_weared_incorrect':
                                mask_incorrect_count += 1
                            else:
                                without_mask_count += 1
                        else:
                            without_mask_count += 1
                    else:
                        without_mask_count += 1
                        
                total_images = len(image_files)
                
        elif dataset_type == "structured":
            with_mask_dir = os.path.join(data_dir, "with_mask")
            without_mask_dir = os.path.join(data_dir, "without_mask")
            incorrect_mask_dir = os.path.join(data_dir, "mask_weared_incorrect")
            
            if os.path.exists(with_mask_dir):
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                with_mask_count = len([f for f in os.listdir(with_mask_dir) if f.lower().endswith(image_extensions)])
                
            if os.path.exists(without_mask_dir):
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                without_mask_count = len([f for f in os.listdir(without_mask_dir) if f.lower().endswith(image_extensions)])
                
            if os.path.exists(incorrect_mask_dir):
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                mask_incorrect_count = len([f for f in os.listdir(incorrect_mask_dir) if f.lower().endswith(image_extensions)])
                
            total_images = with_mask_count + without_mask_count + mask_incorrect_count
            
        elif dataset_type == "loose":
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        without_mask_count += 1
            total_images = without_mask_count
            
        return total_images, with_mask_count, without_mask_count, mask_incorrect_count
    
    def ask_user_for_image_count(self, total_images, with_mask_count, without_mask_count, mask_incorrect_count):
        """Pergunta ao usuário quantas imagens carregar"""
        print(f"\n[INFO] ESTATISTICAS DO DATASET:")
        print(f"   - Total de imagens: {total_images}")
        print(f"   - Com mascara: {with_mask_count}")
        print(f"   - Sem mascara: {without_mask_count}")
        print(f"   - Mascara incorreta: {mask_incorrect_count}")
        
        # Sugere quantidade ideal baseada no dataset
        if total_images >= 1000:
            suggested = min(800, total_images)
        elif total_images >= 500:
            suggested = min(400, total_images) 
        elif total_images >= 200:
            suggested = min(200, total_images)
        else:
            suggested = total_images
            
        # Calcula balanceamento entre as 3 classes
        counts = [with_mask_count, without_mask_count, mask_incorrect_count]
        counts = [c for c in counts if c > 0]  # Remove classes vazias
        
        if len(counts) > 1:
            min_class = min(counts)
            max_class = max(counts)
            balance_ratio = min_class / max_class
        else:
            balance_ratio = 1.0
        
        print(f"\n[SUGESTAO] Para boa acuracia:")
        print(f"   - Recomendado: {suggested} imagens")
        print(f"   - Balanceamento 3 classes: {balance_ratio:.2%} (ideal >70%)")
        
        if balance_ratio < 0.3:
            print(f"[WARNING] Dataset desbalanceado! Considere adicionar mais imagens das classes minoritarias")
            print(f"[INFO] Limitacao: Maximo {min_class * 3} imagens balanceadas ou {total_images} imagens desbalanceadas")
        
        while True:
            try:
                user_input = input(f"\n[PERGUNTA] Quantas imagens carregar? (Enter para {suggested}): ").strip()
                
                if user_input == "":
                    return suggested, "balanced"
                    
                count = int(user_input)
                if count <= 0:
                    print("[ERROR] Numero deve ser maior que 0")
                    continue
                elif count > total_images:
                    print(f"[WARNING] Maximo disponivel: {total_images}")
                    count = total_images
                
                # Pergunta sobre estratégia de carregamento se escolheu número alto
                if count >= min_class * 3:
                    strategy_input = input(f"\n[PERGUNTA] Estrategia de carregamento:\n1. Balanceado (max {min_class * 3} imagens)\n2. Todas as disponiveis ({count} imagens desbalanceadas)\nEscolha (1/2, Enter para 1): ").strip()
                    if strategy_input == "2":
                        return count, "unbalanced"
                    else:
                        return min(count, min_class * 3), "balanced"
                else:
                    return count, "balanced"
                
            except ValueError:
                print("[ERROR] Por favor, digite um numero valido")
                continue
    
    def ask_target_accuracy(self):
        """Pergunta ao usuário qual acurácia deseja atingir"""
        print(f"\n[INFO] META DE ACURACIA:")
        print(f"   - Minimo recomendado: 70% para boa performance")
        print(f"   - Otimo: 85-95% (dependendo do dataset)")
        print(f"   - Maximo teorico: 99% (pode causar overfitting)")
        
        while True:
            try:
                user_input = input(f"\n[PERGUNTA] Qual acuracia deseja atingir? (0-100%, Enter para 80%): ").strip()
                
                if user_input == "":
                    return 0.80  # 80% padrão
                    
                # Remove % se presente
                if user_input.endswith('%'):
                    user_input = user_input[:-1]
                    
                accuracy = float(user_input)
                
                if accuracy <= 0:
                    print("[ERROR] Acuracia deve ser maior que 0")
                    continue
                elif accuracy > 100:
                    print("[ERROR] Acuracia nao pode ser maior que 100%")
                    continue
                elif accuracy > 1:
                    accuracy = accuracy / 100.0  # Converte % para decimal
                    
                if accuracy < 0.5:
                    print("[WARNING] Acuracia muito baixa (<50%). Recomendado minimo 70%")
                elif accuracy > 0.95:
                    print("[WARNING] Acuracia muito alta (>95%). Risco de overfitting")
                    
                return accuracy
                
            except ValueError:
                print("[ERROR] Por favor, digite um numero valido")
                continue

    def load_kaggle_format_data(self, data_dir, max_images=None):
        """Carrega dados no formato Kaggle (images + annotations)"""
        images_dir = os.path.join(data_dir, "images")
        annotations_dir = os.path.join(data_dir, "annotations")
        
        print("[INFO] Carregando dataset formato Kaggle...")
        
        # Lista arquivos de imagem
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
        
        # Se max_images não especificado, pergunta ao usuário
        if max_images is None:
            total_images, with_mask_count, without_mask_count, mask_incorrect_count = self.count_dataset_images(data_dir, "kaggle")
            max_images, loading_strategy = self.ask_user_for_image_count(total_images, with_mask_count, without_mask_count, mask_incorrect_count)
        else:
            loading_strategy = "balanced"
        
        data = []
        labels = []
        
        print(f"[INFO] Estrategia de carregamento: {loading_strategy}")
        
        if loading_strategy == "balanced":
            # Carrega com balanceamento para 3 classes
            with_mask_loaded = 0
            without_mask_loaded = 0
            mask_incorrect_loaded = 0
            max_per_class = max_images // 3  # Divide igualmente entre 3 classes
            print(f"[INFO] Maximo por classe: {max_per_class} imagens")
        else:
            # Carrega sem balanceamento (todas as disponíveis até o limite)
            with_mask_loaded = 0
            without_mask_loaded = 0
            mask_incorrect_loaded = 0
            max_per_class = max_images  # Sem limite por classe
            print(f"[INFO] Carregamento sem balanceamento: ate {max_images} imagens")
        
        for image_file in tqdm(image_files):
            # Verifica se já carregou imagens suficientes
            if len(data) >= max_images:
                break
                
            try:
                # Procura anotação correspondente primeiro para determinar label
                base_name = os.path.splitext(image_file)[0]
                xml_file = os.path.join(annotations_dir, base_name + '.xml')
                
                if not os.path.exists(xml_file):
                    # Se não tem anotação, assume sem máscara como padrão
                    label = 'without_mask'
                else:
                    # Processa anotação XML
                    annotations = self.parse_xml_annotation(xml_file)
                    if annotations:
                        # Usa primeira anotação encontrada
                        label = annotations[0]['label']
                    else:
                        label = 'without_mask'
                
                # Verifica balanceamento apenas se estratégia for "balanced"
                if loading_strategy == "balanced":
                    if label == 'with_mask' and with_mask_loaded >= max_per_class:
                        continue
                    elif label == 'without_mask' and without_mask_loaded >= max_per_class:
                        continue
                    elif label == 'mask_weared_incorrect' and mask_incorrect_loaded >= max_per_class:
                        continue
                
                # Carrega e processa imagem
                image_path = os.path.join(images_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # Preprocessa imagem
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                image = image.astype("float") / 255.0
                
                data.append(image)
                labels.append(label)
                
                # Atualiza contadores de balanceamento
                if label == 'with_mask':
                    with_mask_loaded += 1
                elif label == 'mask_weared_incorrect':
                    mask_incorrect_loaded += 1
                else:
                    without_mask_loaded += 1
                
            except Exception as e:
                print(f"[ERROR] Erro ao processar {image_file}: {e}")
                continue
        
        self.data = np.array(data)
        self.labels = np.array(labels)
        
        # Relatório final de carregamento para 3 classes
        with_mask_final = np.sum(self.labels == 'with_mask')
        without_mask_final = np.sum(self.labels == 'without_mask')
        mask_incorrect_final = np.sum(self.labels == 'mask_weared_incorrect')
        
        counts = [with_mask_final, without_mask_final, mask_incorrect_final]
        counts = [c for c in counts if c > 0]  # Remove classes vazias
        
        if len(counts) > 1:
            min_class = min(counts)
            max_class = max(counts)
            balance_ratio = min_class / max_class
        else:
            balance_ratio = 1.0
        
        print(f"\n[SUCCESS] Dataset Kaggle carregado:")
        print(f"   - Total de imagens: {len(self.data)}")
        print(f"   - Com mascara: {with_mask_final}")
        print(f"   - Sem mascara: {without_mask_final}")
        print(f"   - Mascara incorreta: {mask_incorrect_final}")
        print(f"   - Balanceamento: {balance_ratio:.2%}")
        
        if balance_ratio < 0.7:
            print(f"[WARNING] Dataset ainda desbalanceado (ideal >70%)")
        else:
            print(f"[INFO] Dataset bem balanceado!")
        
        return self.data, self.labels
    
    def check_local_dataset(self):
        """Verifica se existe dataset na pasta data local e determina sua estrutura"""
        data_dir = "data"
        
        if not os.path.exists(data_dir):
            print("[INFO] Pasta 'data' nao encontrada")
            return False, None, None
        
        # Verifica estrutura 1: with_mask / without_mask (estrutura esperada)
        with_mask_dir = os.path.join(data_dir, "with_mask")
        without_mask_dir = os.path.join(data_dir, "without_mask")
        
        if os.path.exists(with_mask_dir) and os.path.exists(without_mask_dir):
            # Conta imagens em cada pasta
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            with_mask_images = [f for f in os.listdir(with_mask_dir) if f.lower().endswith(image_extensions)]
            without_mask_images = [f for f in os.listdir(without_mask_dir) if f.lower().endswith(image_extensions)]
            
            total_images = len(with_mask_images) + len(without_mask_images)
            if total_images > 10:
                print(f"[SUCCESS] Dataset estruturado encontrado: {len(with_mask_images)} com mascara, {len(without_mask_images)} sem mascara")
                return True, data_dir, "structured"
        
        # Verifica estrutura 2: images + annotations (Kaggle format)
        images_dir = os.path.join(data_dir, "images")
        annotations_dir = os.path.join(data_dir, "annotations")
        
        if os.path.exists(images_dir) and os.path.exists(annotations_dir):
            # Conta imagens e anotações
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            xml_extensions = ('.xml',)
            
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
            annotations = [f for f in os.listdir(annotations_dir) if f.lower().endswith(xml_extensions)]
            
            if len(images) > 10 and len(annotations) > 10:
                print(f"[SUCCESS] Dataset Kaggle encontrado: {len(images)} imagens, {len(annotations)} anotacoes XML")
                return True, data_dir, "kaggle"
        
        # Verifica estrutura 3: imagens soltas na pasta data
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        loose_images = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    loose_images.append(os.path.join(root, file))
        
        if len(loose_images) > 10:
            print(f"[SUCCESS] Imagens soltas encontradas: {len(loose_images)} imagens em '{data_dir}'")
            return True, data_dir, "loose"
        
        print(f"[WARNING] Dataset local insuficiente ou mal estruturado")
        return False, None, None

    def load_data(self, dataset_path=None, use_kaggle=True):
        """
        Carrega e prepara os dados para treinamento com prioridade para dataset local
        
        Args:
            dataset_path (str): Caminho para o dataset (opcional)
            use_kaggle (bool): Se deve tentar baixar do Kaggle
        """
        
        # 1. SEMPRE verifica primeiro a pasta 'data' local
        print("[INFO] Verificando pasta 'data' local...")
        has_local, local_path, dataset_type = self.check_local_dataset()
        
        if has_local:
            print(f"[SUCCESS] Usando dataset local: {dataset_type}")
            
            if dataset_type == "kaggle":
                # Dataset com imagens + anotações XML
                return self.load_kaggle_format_data(local_path)
                
            elif dataset_type == "structured":
                # Dataset com pastas with_mask / without_mask
                return self.load_structured_data(local_path)
                
            elif dataset_type == "loose":
                # Imagens soltas - assume sem máscara
                return self.load_loose_images(local_path)
        
        # 2. Se não há dados locais e use_kaggle=True, tenta baixar
        elif use_kaggle:
            print("[INFO] Tentando baixar dataset do Kaggle...")
            try:
                kaggle_path = self.download_dataset()
                if kaggle_path:
                    print("[SUCCESS] Dataset do Kaggle baixado!")
                    # Copia para pasta data local
                    self.copy_to_local_data(kaggle_path)
                    # Recarrega da pasta local
                    has_local, local_path, dataset_type = self.check_local_dataset()
                    if has_local and dataset_type == "kaggle":
                        return self.load_kaggle_format_data(local_path)
                else:
                    print("[WARNING] Falha no download, usando dataset sintetico")
            except Exception as e:
                print(f"[ERROR] Erro no download: {e}, usando dataset sintetico")
        
        # 3. Como último recurso, cria dataset sintético
        print("[WARNING] Usando dataset sintetico para demonstracao")
        dataset_path = self.create_synthetic_dataset()
        return self.load_structured_data(dataset_path)
    
    def copy_to_local_data(self, source_path):
        """Copia dataset baixado para pasta data local"""
        try:
            import shutil
            if os.path.exists("data"):
                shutil.rmtree("data")
            shutil.copytree(source_path, "data")
            print("[SUCCESS] Dataset copiado para pasta 'data' local")
        except Exception as e:
            print(f"[ERROR] Erro ao copiar dataset: {e}")
    
    def load_structured_data(self, data_dir):
        """Carrega dados da estrutura with_mask / without_mask"""
        print("[INFO] Carregando dados estruturados...")
        
        # Categorias
        categories = ["with_mask", "without_mask"]
        
        for category in categories:
            path = os.path.join(data_dir, category)
            if not os.path.exists(path):
                print(f"[WARNING] Diretorio {path} nao encontrado, pulando...")
                continue
                
            label = category
            
            # Lista arquivos de imagem
            image_files = [f for f in os.listdir(path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"[INFO] Processando {len(image_files)} imagens de {category}...")
            
            for filename in tqdm(image_files[:200]):  # Limita para 200 imagens por categoria
                try:
                    img_path = os.path.join(path, filename)
                    image = cv2.imread(img_path)
                    
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                        image = image.astype("float") / 255.0
                        
                        self.data.append(image)
                        self.labels.append(label)
                        
                except Exception as e:
                    print(f"[ERROR] Erro ao processar {filename}: {e}")
        
        # Converte para arrays numpy
        self.data = np.array(self.data, dtype="float32")
        self.labels = np.array(self.labels)
        
        print(f"[SUCCESS] Dados estruturados carregados: {self.data.shape[0]} imagens")
        print(f"   - Com mascara: {np.sum(self.labels == 'with_mask')}")
        print(f"   - Sem mascara: {np.sum(self.labels == 'without_mask')}")
        
        return self.data, self.labels
    
    def load_loose_images(self, data_dir):
        """Carrega imagens soltas (assume todas sem máscara)"""
        print("[INFO] Carregando imagens soltas...")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        data = []
        labels = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    try:
                        img_path = os.path.join(root, file)
                        image = cv2.imread(img_path)
                        
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                            image = image.astype("float") / 255.0
                            
                            data.append(image)
                            labels.append('without_mask')  # Assume sem máscara
                            
                    except Exception as e:
                        print(f"[ERROR] Erro ao processar {file}: {e}")
        
        self.data = np.array(data)
        self.labels = np.array(labels)
        
        print(f"[SUCCESS] Imagens soltas carregadas: {len(self.data)} imagens")
        print(f"[WARNING] Todas classificadas como 'sem mascara' - ajuste manualmente se necessario")
        
        return self.data, self.labels
    
    def prepare_data(self):
        """Prepara os dados para treinamento com 3 classes"""
        print("[INFO] Preparando dados para treinamento...")
        
        # Converte labels para formato categórico com 3 classes
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded_labels = le.fit_transform(self.labels)
        self.labels = to_categorical(encoded_labels, num_classes=3)
        
        # Armazena mapeamento para usar na avaliação
        self.label_encoder = le
        
        # Divide em treino e teste
        (trainX, testX, trainY, testY) = train_test_split(
            self.data, self.labels,
            test_size=0.20, stratify=self.labels, random_state=42
        )
        
        print(f"[SUCCESS] Dados divididos:")
        print(f"   - Treino: {trainX.shape[0]} imagens")
        print(f"   - Teste: {testX.shape[0]} imagens")
        
        return trainX, testX, trainY, testY
    
    def create_model(self):
        """Cria a arquitetura do modelo CNN"""
        print("[INFO] Criando modelo CNN...")
        
        model = Sequential([
            # Input layer
            Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3)),
            
            # Primeiro bloco convolucional
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Segundo bloco convolucional
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Terceiro bloco convolucional
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Quarto bloco convolucional
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Camadas densas
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(3, activation='softmax')  # 3 classes: com_mascara/sem_mascara/mascara_incorreta
        ])
        
        # Compila o modelo
        opt = Adam(learning_rate=1e-4)
        model.compile(
            loss="categorical_crossentropy",  # Para 3 classes com softmax
            optimizer=opt,
            metrics=["accuracy"]
        )
        
        self.model = model
        print("[SUCCESS] Modelo criado!")
        print(f"[INFO] Total de parametros: {model.count_params():,}")
        
        return model
    
    def calculate_class_weights(self, labels):
        """Calcula pesos das classes para balanceamento"""
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Converte one-hot para labels
        if len(labels.shape) > 1:
            label_indices = np.argmax(labels, axis=1)
        else:
            label_indices = labels
            
        classes = np.unique(label_indices)
        class_weights = compute_class_weight('balanced', classes=classes, y=label_indices)
        
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"[INFO] Pesos das classes calculados:")
        for class_idx, weight in class_weight_dict.items():
            if hasattr(self, 'label_encoder') and class_idx < len(self.label_encoder.classes_):
                class_name = self.label_encoder.classes_[class_idx]
                print(f"   - {class_name}: {weight:.2f}")
            else:
                print(f"   - Classe {class_idx}: {weight:.2f}")
                
        return class_weight_dict

    def train_model(self, trainX, trainY, testX, testY, target_accuracy=0.80, max_epochs=100):
        """
        Treina o modelo até atingir acurácia alvo
        
        Args:
            trainX, trainY: Dados de treino
            testX, testY: Dados de teste
            target_accuracy: Acurácia alvo (0.0 a 1.0)
            max_epochs: Máximo de épocas
        """
        print(f"[INFO] Treinamento com meta de acuracia: {target_accuracy:.1%}")
        print(f"[INFO] Maximo de epocas: {max_epochs}")
        
        # Calcula pesos das classes para balanceamento
        class_weights = self.calculate_class_weights(trainY)
        
        # Data augmentation intensivo para melhorar acurácia
        aug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest"
        )
        
        # Callback personalizado para parar quando atingir meta
        from tensorflow.keras.callbacks import Callback
        
        class AccuracyTarget(Callback):
            def __init__(self, target_acc):
                self.target_acc = target_acc
                
            def on_epoch_end(self, epoch, logs=None):
                val_acc = logs.get('val_accuracy')
                if val_acc and val_acc >= self.target_acc:
                    print(f"\n[SUCCESS] Meta de acuracia atingida: {val_acc:.1%} >= {self.target_acc:.1%}")
                    self.model.stop_training = True
        
        # Learning rate scheduler
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 30:
                return lr * 0.5
            else:
                return lr * 0.1
                
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                "model/mask_detector.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Treina o modelo
        history = self.model.fit(
            aug.flow(trainX, trainY, batch_size=self.BATCH_SIZE),
            steps_per_epoch=len(trainX) // self.BATCH_SIZE,
            validation_data=(testX, testY),
            validation_steps=len(testX) // self.BATCH_SIZE,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("[SUCCESS] Treinamento concluido!")
        return history
    
    def evaluate_model(self, testX, testY):
        """Avalia o modelo nos dados de teste com 3 classes"""
        print("[INFO] Avaliando modelo...")
        
        predictions = self.model.predict(testX, batch_size=self.BATCH_SIZE)
        predictions = np.argmax(predictions, axis=1)
        testY_labels = np.argmax(testY, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Define nomes das classes baseado no label encoder
        if hasattr(self, 'label_encoder'):
            class_names = self.label_encoder.classes_
        else:
            class_names = ["mask_weared_incorrect", "with_mask", "without_mask"]  # Ordem padrão alfabética
        
        print("[INFO] Relatorio de classificacao:")
        print(classification_report(testY_labels, predictions, 
                                  target_names=class_names))
        
        print("🔢 Matriz de confusão:")
        print(confusion_matrix(testY_labels, predictions))
        
        # Mostra distribuição de classes
        unique, counts = np.unique(testY_labels, return_counts=True)
        print("\n[INFO] Distribuição das classes no teste:")
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            if class_idx < len(class_names):
                print(f"   - {class_names[class_idx]}: {count} imagens")
    
    def plot_training_history(self, history):
        """Plota gráficos do histórico de treinamento"""
        print("[INFO] Plotando historico de treinamento...")
        
        # Gráfico de acurácia
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Treino')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title('Acurácia do Modelo')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        
        # Gráfico de perda
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title('Perda do Modelo')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("[SUCCESS] Graficos salvos em model/training_history.png")

def main():
    """Função principal para treinar o modelo"""
    print("[INFO] Treinador de Modelo de Deteccao de Mascaras")
    print("=" * 50)
    
    # Cria diretório do modelo
    os.makedirs("model", exist_ok=True)
    
    # Inicializa o treinador
    trainer = MaskModelTrainer(img_size=224, batch_size=32)
    
    try:
        # Tenta baixar dataset real
        dataset_path = trainer.download_dataset()
    except:
        dataset_path = None
    
    # Carrega dados
    trainer.load_data(dataset_path)
    
    if len(trainer.data) == 0:
        print("[ERROR] Nenhum dado carregado. Verifique o dataset.")
        return
    
    # Prepara dados
    trainX, testX, trainY, testY = trainer.prepare_data()
    
    # Pergunta meta de acurácia
    target_accuracy = trainer.ask_target_accuracy()
    
    # Cria modelo
    trainer.create_model()
    
    # Treina modelo até atingir meta
    history = trainer.train_model(trainX, trainY, testX, testY, target_accuracy=target_accuracy)
    
    # Avalia modelo
    trainer.evaluate_model(testX, testY)
    
    # Plota histórico
    trainer.plot_training_history(history)
    
    # Salva informações do modelo
    model_info = {
        "img_size": trainer.IMG_SIZE,
        "classes": ["mask_weared_incorrect", "with_mask", "without_mask"],
        "total_images": len(trainer.data),
        "accuracy": max(history.history['val_accuracy']),
        "target_accuracy": target_accuracy,
        "model_path": "model/mask_detector.h5"
    }
    
    with open("model/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("\n[SUCCESS] Treinamento concluido com sucesso!")
    print(f"📁 Modelo salvo em: model/mask_detector.h5")
    print(f"[INFO] Acuracia final: {model_info['accuracy']:.4f}")

if __name__ == "__main__":
    main()