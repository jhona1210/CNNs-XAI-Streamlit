import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
from PIL import Image

@st.cache_resource
def load_trained_model(model_path):
    """Carga el modelo Keras. Se cachea para no recargar en cada interacción."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_file, target_size=(150, 150)):
    """Lee y preprocesa la imagen de Streamlit para el modelo."""
    # Convertir a imagen de PIL
    img = Image.open(image_file)
    
    # Asegurar que está en RGB (por si es RGBA o B/N)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Redimensionar
    img = img.resize(target_size)
    
    # Convertir a array de numpy
    img_array = np.array(img, dtype=np.float32)
    
    # Expandir dimensiones (batch_size, 150, 150, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Normalizar
    img_array_expanded /= 255.0
    
    return img_array_expanded, np.array(img)

def predict_image(model, img_array):
    """Realiza la predicción y devuelve la clase y probabilidad."""
    prediccion = model.predict(img_array)[0][0]
    
    # class_idx = 1 (Male), class_idx = 0 (Female) basado en tu código
    if prediccion > 0.5:
        clase = "Hombre"
        probabilidad = prediccion
        class_idx = 1
    else:
        clase = "Mujer"
        probabilidad = 1.0 - prediccion
        class_idx = 0
        
    return clase, probabilidad, class_idx

# =============================================
# FUNCIONES XAI (Basadas en tu notebook)
# =============================================

def compute_saliency_map(model, img_preprocessed, class_idx=0):
    """Calcula el Saliency Map. El class_idx para el gradiente es 0 porque solo hay 1 salida en el modelo (binario)."""
    img_tensor = tf.convert_to_tensor(img_preprocessed, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor, training=False)
        target_score = predictions[:, 0]  # Siempre 0 para clasificación binaria con 1 nodo
        
    grads = tape.gradient(target_score, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
    
    # Para visualización en streamlit, pasamos el mapa a uint8
    saliency_uint8 = np.uint8(255 * saliency.numpy().squeeze())
    # Aplicar un mapa de color 'hot' con cv2
    saliency_colored = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_HOT)
    # OpenCV usa BGR, convertimos a RGB para Streamlit
    saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
    
    return saliency_colored

def find_last_conv_layer(model):
    """Encuentra la última capa convolucional del modelo."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No se encontró ninguna capa convolucional.")

def compute_gradcam(model, img_preprocessed, last_conv_layer_name):
    """Calcula el heatmap de Grad-CAM."""
    img_input = tf.keras.Input(shape=img_preprocessed.shape[1:])
    
    x = img_input
    target_layer_output = None
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            target_layer_output = x
            
    if target_layer_output is None:
        raise ValueError(f"Capa {last_conv_layer_name} no encontrada.")
        
    grad_model = tf.keras.Model(inputs=img_input, outputs=[target_layer_output, x])
    
    img_tensor = tf.convert_to_tensor(img_preprocessed, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        target_score = predictions[:, 0]  # Siempre 0 en binario con salida única
        
    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + 1e-8)
    
    return heatmap.numpy()

def superimpose_heatmap(original_img, heatmap, alpha=0.6, beta=0.4):
    """Superpone el heatmap de Grad-CAM en la imagen original."""
    if original_img.max() <= 1.0:
        original_display = (original_img * 255).astype(np.uint8)
    else:
        original_display = original_img.astype(np.uint8)
        
    heatmap_resized = cv2.resize(heatmap, (original_display.shape[1], original_display.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # OpenCV colormap da BGR, convertimos a RGB para sumar correctamente con la imagen original (que debe ser RGB)
    # Original_img ya está en RGB según PIL.
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = cv2.addWeighted(original_display, alpha, heatmap_colored_rgb, beta, 0)
    
    return superimposed
