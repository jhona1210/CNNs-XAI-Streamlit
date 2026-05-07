import streamlit as st
import xai_utils
import numpy as np

# ==========================================
# 1. Configuración y Estilos de la Página
# ==========================================
st.set_page_config(page_title="CNN Gender Classifier & XAI", page_icon="🧠", layout="wide")

# Inyectando CSS personalizado con la paleta solicitada
# Colores: #0E3770, #14253D, #6F9AD6, #B8D5FF, #E0EDFF
st.markdown("""
<style>
    /* Fondo principal */
    .stApp {
        background-color: #14253D;
        color: #E0EDFF;
    }
    
    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #B8D5FF !important;
    }
    
    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #0E3770;
    }
    
    /* Botones */
    .stButton>button {
        background-color: #6F9AD6;
        color: #14253D;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #B8D5FF;
        color: #14253D;
    }
    
    /* Valores de métricas */
    [data-testid="stMetricValue"] {
        color: #B8D5FF;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Carga del Modelo
# ==========================================
MODEL_PATH = "models/modelo_caras_entrenado_keras_2.keras"

@st.cache_resource
def load_model():
    return xai_utils.load_trained_model(MODEL_PATH)

try:
    modelo = load_model()
    st.sidebar.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================================
# 3. Interfaz Principal
# ==========================================
st.title("Clasificador de Género y Visualización XAI 🧠")
st.markdown("""
Esta aplicación permite predecir el género (Hombre/Mujer) de un rostro utilizando una **Red Neuronal Convolucional (CNN)**.
Además, utiliza técnicas de **Interpretabilidad Visual (XAI)** como Saliency Maps y Grad-CAM para mostrar 
qué partes de la imagen fueron relevantes para la decisión del modelo.
""")

st.header("1. Sube una imagen de un rostro")
uploaded_file = st.file_uploader("Selecciona una imagen en formato JPG o PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostramos mensaje de procesamiento
    with st.spinner('Procesando imagen y ejecutando modelo...'):
        
        # 4. Preprocesamiento y Predicción
        img_preprocesada, img_original = xai_utils.preprocess_image(uploaded_file, target_size=(150, 150))
        
        # Ojo: esto es un workaround para Keras 3 (asegurar que la entrada se registra)
        _ = modelo(img_preprocesada)
        
        clase_predicha, probabilidad, _ = xai_utils.predict_image(modelo, img_preprocesada)
        
        # 5. Generar Mapas XAI
        # Saliency Map
        saliency_map = xai_utils.compute_saliency_map(modelo, img_preprocesada)
        
        # Grad-CAM
        last_conv = xai_utils.find_last_conv_layer(modelo)
        heatmap = xai_utils.compute_gradcam(modelo, img_preprocesada, last_conv)
        gradcam_img = xai_utils.superimpose_heatmap(img_original, heatmap)
        
    st.success("¡Análisis completado!")
    
    # Mostrar resultados de clasificación
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Clase Predicha", value=clase_predicha)
    with col_res2:
        st.metric(label="Confianza (Probabilidad)", value=f"{probabilidad*100:.2f} %")

    # Mostrar las 3 imágenes
    st.header("2. Resultados Visuales e Interpretabilidad")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(img_original, use_container_width=True)
        st.info("Imagen redimensionada a 150x150 como lo espera la red.")
        
    with col2:
        st.subheader("Saliency Map")
        st.image(saliency_map, use_container_width=True)
        st.info("Resalta los píxeles individuales más importantes para la red.")
        
    with col3:
        st.subheader("Grad-CAM")
        st.image(gradcam_img, use_container_width=True)
        st.info(f"Muestra las regiones más activadas en la capa {last_conv}.")
