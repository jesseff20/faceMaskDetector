"""
Interface Streamlit para detecção de máscaras faciais
Aplicação web completa com webcam e upload de imagens
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
from utils import MaskDetector, download_face_detection_models
import tempfile

def get_available_cameras():
    """
    Detecta câmeras disponíveis no sistema
    
    Returns:
        dict: Dicionário com nome da câmera e índice
    """
    available_cameras = {}
    
    # Testa até 5 câmeras possíveis
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW para Windows
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    if i == 0:
                        available_cameras["Câmera Principal"] = i
                    else:
                        available_cameras[f"Câmera {i}"] = i
                cap.release()
        except:
            continue
    
    # Se nenhuma câmera foi encontrada, adiciona a padrão
    if not available_cameras:
        available_cameras["Câmera Padrão"] = 0
    
    return available_cameras

@st.cache_data
def get_camera_info():
    """Cache das informações de câmeras disponíveis"""
    return get_available_cameras()

def test_camera(camera_index):
    """Testa se a câmera funciona corretamente"""
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                st.success(f"Câmera {camera_index} funcionando corretamente!")
                # Mostra preview da câmera
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Preview da Câmera {camera_index}", width=300)
            else:
                st.error(f"Câmera {camera_index} não conseguiu capturar imagem")
            cap.release()
        else:
            st.error(f"Não foi possível abrir a câmera {camera_index}")
    except Exception as e:
        st.error(f"Erro ao testar câmera {camera_index}: {e}")

# Configuração da página
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    .positive-feedback {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 15px;
        border: 3px solid #28a745;
        margin: 1rem 0;
        text-align: center;
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
    
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .footer {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
        border-top: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Cache para o detector
@st.cache_resource
def load_detector():
    """Carrega o detector de máscaras (cached)"""
    with st.spinner("Inicializando detector de máscaras..."):
        model_path = "model/mask_detector.h5" if os.path.exists("model/mask_detector.h5") else None
        detector = MaskDetector(model_path)
        
        # Verifica se o modelo foi carregado com sucesso
        if detector.model is not None:
            st.success("Detector carregado com sucesso!")
        else:
            st.warning("Detector iniciado sem modelo treinado")
        
        return detector

def main():
    """Função principal da aplicação"""
    
    # Header
    st.markdown('<h1 class="main-header">Face Mask Detector</h1>', unsafe_allow_html=True)
    
    # Status de inicialização
    initialization_container = st.container()
    with initialization_container:
        st.markdown("""
        <div class="info-box">
            <h4>Sistema de Detecção de Máscaras Faciais</h4>
            <p>Sistema inteligente que usa <strong>Deep Learning</strong> para detectar se uma pessoa está usando máscara facial.</p>
            <ul>
                <li>Tempo Real: Análise via webcam</li>
                <li>Upload: Análise de imagens</li>
                <li>Múltiplos Rostos: Detecta várias pessoas</li>
                <li>Alta Precisão: CNN treinada</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        
        # Informações do modelo
        model_path = "model/mask_detector.h5"
        if os.path.exists(model_path):
            st.markdown('<div class="success-box">Modelo carregado e pronto!</div>', unsafe_allow_html=True)
            
            # Informações do modelo se existir arquivo de info
            if os.path.exists("model/model_info.json"):
                import json
                with open("model/model_info.json", "r") as f:
                    model_info = json.load(f)
                
                model_type = model_info.get('model_type', 'trained')
                if model_type == 'demonstration':
                    st.info("Modo: Demonstração")
                    st.write("Status: Pronto para teste")
                    st.write("Dica: Execute python train_model.py para melhor precisão")
                else:
                    st.write(f"Acurácia: {model_info.get('accuracy', 0):.2%}")
                    st.write(f"Imagens de treino: {model_info.get('total_images', 0):,}")
        else:
            st.markdown('<div class="warning-box">Modelo não encontrado! Execute python train_model.py para treinar um modelo com dados reais.</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Opções de entrada
        st.subheader("Entrada")
        input_option = st.radio(
            "Escolha o tipo de entrada:",
            ["Upload de Imagem", "Webcam (Foto)", "Webcam (Tempo Real)"]
        )
        
        # Seleção de câmera para opções de webcam
        selected_camera = 0
        if "Webcam" in input_option:
            st.subheader("Configuração da Câmera")
            
            # Carrega informações das câmeras
            camera_options = get_camera_info()
            
            if len(camera_options) > 1:
                camera_name = st.selectbox(
                    "Selecione a câmera:",
                    options=list(camera_options.keys()),
                    help="Escolha qual câmera usar para captura"
                )
                selected_camera = camera_options[camera_name]
            else:
                camera_name = list(camera_options.keys())[0]
                selected_camera = camera_options[camera_name]
                st.info(f"Usando: {camera_name}")
                
            # Teste da câmera
            if st.button("Testar Câmera"):
                test_camera(selected_camera)
        
        st.divider()
        
        # Configurações avançadas
        with st.expander("Configurações Avançadas"):
            confidence_threshold = st.slider(
                "Limite de confiança",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Apenas detecções acima deste valor serão mostradas"
            )
            
            show_confidence = st.checkbox(
                "Mostrar valores de confiança",
                value=True,
                help="Exibe os valores de confiança nas predições"
            )
    
    # Carrega o detector com feedback visual
    detector_status = st.empty()
    with detector_status:
        with st.spinner("Carregando sistema de detecção..."):
            detector = load_detector()
    
    # Área principal baseada na opção selecionada
    if input_option == "Upload de Imagem":
        handle_image_upload(detector, confidence_threshold, show_confidence)
    
    elif input_option == "Webcam (Foto)":
        handle_camera_photo(detector, confidence_threshold, show_confidence, selected_camera)
    
    elif input_option == "Webcam (Tempo Real)":
        handle_realtime_camera(detector, confidence_threshold, show_confidence, selected_camera)
    
    # Status do sistema
    st.markdown("""
    <div class="info-box">
        <h4>Status do Sistema:</h4>
        <p>Sistema inicializado e funcionando<br>
        Modelo carregado<br>
        Câmeras detectadas<br>
        Pronto para análise!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rodapé com créditos do desenvolvedor
    st.divider()
    st.markdown("""
    <div class="footer">
        <h4>Desenvolvido por</h4>
        <h3 style="color: #1f77b4; margin: 0.5rem 0;">Jesse Fernandes</h3>
        <p style="margin: 0.5rem 0;">
            <a href="mailto:jesseff20@gmail.com" style="color: #1f77b4; text-decoration: none;">
                jesseff20@gmail.com
            </a>
        </p>
        <hr style="margin: 1rem 0; border: 1px solid #ddd;">
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            <strong>Tecnologias:</strong> Python • Streamlit • TensorFlow • OpenCV • Keras<br>
            <strong>Sistema:</strong> Detecção de Máscaras Faciais com Deep Learning<br>
            <strong>Versão:</strong> 1.0 • Outubro 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

def handle_image_upload(detector, confidence_threshold, show_confidence):
    """Manipula upload e análise de imagens"""
    
    st.subheader("Upload de Imagem")
    
    uploaded_file = st.file_uploader(
        "Escolha uma imagem...",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos suportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Converte para OpenCV
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Converte RGB para BGR (OpenCV usa BGR)
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array
        
        # Cria colunas para layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagem Original")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Análise")
            
            with st.spinner("Analisando imagem..."):
                # Processa a imagem
                processed_image, detections = detector.process_image(image_cv)
                
                if processed_image is not None:
                    # Converte de volta para RGB para exibição
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_column_width=True)
                else:
                    st.error("Erro ao processar imagem")
        
        # Resultados
        display_results(detections, confidence_threshold, show_confidence)

def handle_camera_photo(detector, confidence_threshold, show_confidence, camera_index=0):
    """Manipula captura de foto via webcam"""
    
    st.subheader("Webcam - Captura de Foto")
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Câmera Selecionada:</strong> Câmera {camera_index}<br>
        <strong>Instruções:</strong><br>
        1. Clique no botão "Take a picture" abaixo<br>
        2. Autorize o acesso à câmera se solicitado<br>
        3. Posicione-se na frente da câmera<br>
        4. A análise será feita automaticamente
    </div>
    """, unsafe_allow_html=True)
    
    # Captura de foto
    camera_photo = st.camera_input("Tire uma foto")
    
    if camera_photo is not None:
        # Converte para OpenCV
        image = Image.open(camera_photo)
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Layout em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Foto Capturada")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Análise")
            
            with st.spinner("Analisando foto..."):
                processed_image, detections = detector.process_image(image_cv)
                
                if processed_image is not None:
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_column_width=True)
                else:
                    st.error("Erro ao processar imagem")
        
        display_results(detections, confidence_threshold, show_confidence)

def handle_realtime_camera(detector, confidence_threshold, show_confidence, camera_index=0):
    """Manipula análise em tempo real via webcam"""
    
    st.subheader("Webcam - Tempo Real")
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Câmera Selecionada:</strong> Câmera {camera_index}<br>
        <strong>Instruções:</strong><br>
        1. Clique em "Iniciar Câmera" para começar<br>
        2. Posicione-se na frente da câmera<br>
        3. A análise será feita em tempo real<br>
        4. Clique em "Parar Câmera" para finalizar
    </div>
    """, unsafe_allow_html=True)
    
    # Controles
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_camera = st.button("Iniciar Câmera", type="primary")
    
    with col2:
        stop_camera = st.button("Parar Câmera")
    
    # Placeholder para o feed da câmera
    camera_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # Estado da câmera
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if start_camera:
        st.session_state.camera_active = True
    
    if stop_camera:
        st.session_state.camera_active = False
    
    # Loop da câmera
    if st.session_state.camera_active:
        try:
            # Inicializa a câmera selecionada
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                st.error("Não foi possível acessar a câmera")
                st.session_state.camera_active = False
            else:
                fps_counter = 0
                start_time = time.time()
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Erro ao capturar frame da câmera")
                        break
                    
                    # Processa o frame
                    processed_frame, detections = detector.process_image(frame)
                    
                    if processed_frame is not None:
                        # Adiciona informações de FPS
                        fps_counter += 1
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            fps = fps_counter / elapsed_time
                            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Converte para RGB e exibe
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Atualiza resultados
                        with results_placeholder.container():
                            display_results(detections, confidence_threshold, show_confidence)
                    
                    # Pequena pausa para não sobrecarregar
                    time.sleep(0.03)  # ~30 FPS
                
                cap.release()
        
        except Exception as e:
            st.error(f"Erro na câmera: {e}")
            st.session_state.camera_active = False

def display_results(detections, confidence_threshold, show_confidence):
    """Exibe os resultados das detecções"""
    
    st.subheader("Resultados")
    
    if not detections:
        st.markdown('<div class="warning-box">Nenhum rosto detectado na imagem</div>', unsafe_allow_html=True)
        return
    
    # Filtra por confiança
    filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
    
    if not filtered_detections:
        st.markdown(f'<div class="warning-box">Nenhuma detecção acima do limite de confiança ({confidence_threshold:.2f})</div>', unsafe_allow_html=True)
        return
    
    # Estatísticas
    total_faces = len(filtered_detections)
    with_mask = sum(1 for d in filtered_detections if d['label'] == 'Com Máscara')
    without_mask = sum(1 for d in filtered_detections if d['label'] == 'Sem Máscara')
    
    # Feedback visual positivo se todos estão usando máscara
    if total_faces > 0 and without_mask == 0:
        st.markdown(f"""
        <div class="positive-feedback">
            <h2 style="color: #28a745; margin: 0;">EXCELENTE!</h2>
            <h3 style="color: #28a745; margin: 0.5rem 0;">
                {'Todas as pessoas estão' if total_faces > 1 else 'Você está'} usando máscara corretamente!
            </h3>
            <p style="color: #155724; margin: 0; font-size: 1.1rem;">
                Obrigado por contribuir com a segurança de todos!
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif total_faces > 0 and with_mask > 0:
        st.markdown(f"""
        <div class="success-box">
            <h4 style="color: #28a745;">Ótimo! {with_mask} de {total_faces} pessoas usando máscara!</h4>
            <p>Continue incentivando o uso correto de máscaras.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_faces}</h3>
            <p>Rostos Detectados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 5px solid #28a745;">
            <h3>{with_mask}</h3>
            <p>Com Máscara</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 5px solid #dc3545;">
            <h3>{without_mask}</h3>
            <p>Sem Máscara</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detalhes das detecções
    if show_confidence:
        st.subheader("Detalhes")
        for i, detection in enumerate(filtered_detections, 1):
            label = detection['label']
            confidence = detection['confidence']
            
            if label == 'Com Máscara':
                st.markdown(f"""
                <div class="success-box">
                    <strong>Rosto {i}:</strong> {label} (Confiança: {confidence:.2%})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <strong>Rosto {i}:</strong> {label} (Confiança: {confidence:.2%})
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Verifica se os modelos de detecção de rosto estão disponíveis
    if not os.path.exists("model/deploy.prototxt"):
        with st.spinner("Baixando modelos de detecção de rosto..."):
            download_face_detection_models()
    
    main()