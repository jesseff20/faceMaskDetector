"""
Utilitários para detecção de máscaras faciais
Contém funções para detecção de rostos, carregamento de modelos e classificação
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st

class MaskDetector:
    def __init__(self, model_path=None):
        """
        Inicializa o detector de máscaras
        
        Args:
            model_path (str): Caminho para o modelo treinado
        """
        self.model = None
        self.face_net = None
        self.face_cascade = None
        self.face_cascade_profile = None
        self.model_path = model_path
        self.load_face_detector()
        
        if model_path and os.path.exists(model_path):
            self.load_mask_model(model_path)
    
    def load_face_detector(self):
        """Carrega múltiplos detectores de rostos para maior precisão"""
        try:
            # Carrega detector frontal padrão
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Carrega detector de perfil
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.face_cascade_profile = cv2.CascadeClassifier(profile_cascade_path)
            
            # Carrega detector DNN (mais preciso)
            prototxt_path = "model/deploy.prototxt"
            weights_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(prototxt_path) and os.path.exists(weights_path):
                self.face_net = cv2.dnn.readNet(prototxt_path, weights_path)
                print("✅ Detector de rostos DNN carregado!")
            else:
                print("ℹ️ Usando Haar Cascades para detecção de rostos")
                
            print("✅ Detectores múltiplos carregados (frontal + perfil)")
                
        except Exception as e:
            print(f"❌ Erro ao carregar detectores de rostos: {e}")
            self.face_cascade = None
            self.face_cascade_profile = None
            self.face_net = None
    
    def load_mask_model(self, model_path):
        """
        Carrega o modelo de detecção de máscaras
        
        Args:
            model_path (str): Caminho para o arquivo do modelo (.h5)
        """
        try:
            self.model = load_model(model_path)
            print(f"✅ Modelo de máscaras carregado: {model_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
    
    def detect_faces(self, image):
        """
        Detecta rostos na imagem usando múltiplos detectores e escalas
        
        Args:
            image: Imagem de entrada (array numpy)
            
        Returns:
            list: Lista de coordenadas dos rostos [(x, y, w, h), ...]
        """
        all_faces = []
        
        # 1. Tenta DNN primeiro (mais preciso)
        if self.face_net is not None:
            dnn_faces = self._detect_faces_dnn(image)
            all_faces.extend(dnn_faces)
        
        # 2. Usa Haar Cascade frontal com múltiplas escalas
        if self.face_cascade is not None:
            frontal_faces = self._detect_faces_haar_multi_scale(image)
            all_faces.extend(frontal_faces)
        
        # 3. Usa Haar Cascade para perfil
        if self.face_cascade_profile is not None:
            profile_faces = self._detect_faces_profile(image)
            all_faces.extend(profile_faces)
        
        # Remove duplicatas usando Non-Maximum Suppression
        faces = self._remove_duplicate_faces(all_faces)
        
        return faces
    
    def _detect_faces_haar_multi_scale(self, image):
        """Detecta rostos usando Haar Cascade com múltiplas escalas"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplica equalização de histograma para melhor detecção
        gray = cv2.equalizeHist(gray)
        
        faces = []
        
        # Múltiplas escalas para detectar rostos de diferentes tamanhos
        scale_factors = [1.05, 1.1, 1.2, 1.3]
        min_neighbors_values = [3, 4, 5, 6]
        min_sizes = [(20, 20), (30, 30), (40, 40), (50, 50)]
        
        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_values:
                for min_size in min_sizes:
                    detected = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=min_size,
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(detected) > 0:
                        faces.extend(detected.tolist())
        
        return faces
    
    def _detect_faces_profile(self, image):
        """Detecta rostos de perfil"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = []
        
        # Detecção de perfil esquerdo
        profile_faces = self.face_cascade_profile.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(profile_faces) > 0:
            faces.extend(profile_faces.tolist())
        
        # Detecção de perfil direito (imagem espelhada)
        flipped = cv2.flip(gray, 1)
        flipped_faces = self.face_cascade_profile.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Converte coordenadas de volta para imagem original
        if len(flipped_faces) > 0:
            for (x, y, w, h) in flipped_faces:
                x_original = gray.shape[1] - x - w
                faces.append([x_original, y, w, h])
        
        return faces
    
    def _remove_duplicate_faces(self, faces):
        """Remove faces duplicadas usando Non-Maximum Suppression"""
        if not faces:
            return []
        
        faces = np.array(faces)
        
        # Converte (x, y, w, h) para (x1, y1, x2, y2)
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        boxes = np.array(boxes, dtype=np.float32)
        
        # Aplica Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            [1.0] * len(boxes),  # Todas com confiança 1.0
            score_threshold=0.5,
            nms_threshold=0.3
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i].tolist() for i in indices]
        else:
            return faces.tolist()
    
    def _detect_faces_dnn(self, image):
        """Detecta rostos usando DNN"""
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append([x, y, x1-x, y1-y])
        
        return faces
    
    def predict_mask(self, face_image):
        """
        Prediz se há máscara no rosto
        
        Args:
            face_image: Imagem do rosto (array numpy)
            
        Returns:
            tuple: (label, confidence) onde label é 'Com Máscara' ou 'Sem Máscara'
        """
        if self.model is None:
            return "Modelo não carregado", 0.0
        
        try:
            # Pré-processa a imagem
            face_image = cv2.resize(face_image, (224, 224))
            face_image = img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = face_image / 255.0
            
            # Faz a predição
            prediction = self.model.predict(face_image, verbose=0)[0]
            
            # Interpreta o resultado
            if len(prediction) == 2:
                # Modelo binário: [sem_mascara, com_mascara]
                mask_prob = prediction[1]
                no_mask_prob = prediction[0]
            else:
                # Modelo com uma saída (sigmoid)
                mask_prob = prediction[0]
                no_mask_prob = 1 - mask_prob
            
            if mask_prob > no_mask_prob:
                return "Com Máscara", float(mask_prob)
            else:
                return "Sem Máscara", float(no_mask_prob)
                
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            return "Erro", 0.0
    
    def process_image(self, image):
        """
        Processa uma imagem completa, detectando rostos e classificando máscaras
        
        Args:
            image: Imagem de entrada
            
        Returns:
            tuple: (imagem_processada, detecções)
        """
        if image is None:
            return None, []
        
        # Copia a imagem original
        processed_image = image.copy()
        detections = []
        
        # Detecta rostos
        faces = self.detect_faces(image)
        
        for (x, y, w, h) in faces:
            # Extrai o rosto
            face = image[y:y+h, x:x+w]
            
            if face.size > 0:
                # Prediz máscara
                label, confidence = self.predict_mask(face)
                
                # Define cores
                if label == "Com Máscara":
                    color = (0, 255, 0)  # Verde
                else:
                    color = (0, 0, 255)  # Vermelho
                
                # Desenha retângulo e label
                cv2.rectangle(processed_image, (x, y), (x+w, y+h), color, 2)
                
                # Prepara o texto
                text = f"{label}: {confidence:.2f}"
                
                # Calcula posição do texto
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Desenha fundo do texto
                cv2.rectangle(
                    processed_image, 
                    (x, y - text_height - 10), 
                    (x + text_width, y), 
                    color, -1
                )
                
                # Desenha o texto
                cv2.putText(
                    processed_image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Adiciona à lista de detecções
                detections.append({
                    'bbox': (x, y, w, h),
                    'label': label,
                    'confidence': confidence
                })
        
        return processed_image, detections

def download_face_detection_models():
    """
    Baixa os modelos DNN para detecção de rostos se não existirem
    """
    import requests
    import os
    
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    files_to_download = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            print(f"📥 Baixando {filename}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✅ {filename} baixado!")
            except Exception as e:
                print(f"❌ Erro ao baixar {filename}: {e}")



@st.cache_resource
def load_mask_detector(model_path=None):
    """
    Carrega o detector de máscaras (cached para Streamlit)
    
    Args:
        model_path (str): Caminho para o modelo
        
    Returns:
        MaskDetector: Instância do detector
    """
    return MaskDetector(model_path)

def preprocess_image(image, target_size=(224, 224)):
    """
    Pré-processa imagem para o modelo
    
    Args:
        image: Imagem de entrada
        target_size: Tamanho desejado (largura, altura)
        
    Returns:
        numpy.array: Imagem pré-processada
    """
    if image is None:
        return None
    
    # Redimensiona
    image = cv2.resize(image, target_size)
    
    # Normaliza
    image = image.astype(np.float32) / 255.0
    
    return image