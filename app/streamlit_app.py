import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import img_to_array

# Ajusta caminhos para executar com: streamlit run app/streamlit_app.py
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
CLASS_INDICES_PATH = BASE_DIR / "models" / "class_indices.json"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "reports" / "confusion_matrix.png"
IMAGE_SIZE = (64, 64)


@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file:
        class_indices = json.load(file)
    return {value: key for key, value in class_indices.items()}


def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    resized_image = image.resize(IMAGE_SIZE)
    image_array = img_to_array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array


st.set_page_config(page_title="Classificador CNN", page_icon="🧠", layout="centered")

st.title("Classificador de Imagens com CNN")
st.write("Aplicação simples para testar o modelo CNN treinado localmente.")

if not MODEL_PATH.exists() or not CLASS_INDICES_PATH.exists():
    st.warning("Modelo não encontrado. Execute primeiro: `python src/train.py`")
    st.stop()

model = load_trained_model()
class_names = load_class_names()

uploaded_file = st.file_uploader(
    "Envie uma imagem para classificação",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original_image, processed_image = preprocess_image(uploaded_file)

    st.image(original_image, caption="Imagem enviada", use_container_width=True)

    probability = float(model.predict(processed_image)[0][0])
    predicted_index = 1 if probability > 0.5 else 0
    predicted_class = class_names[predicted_index]
    confidence = probability if predicted_index == 1 else 1 - probability

    st.subheader("Resultado")
    st.success(f"Classe prevista: **{predicted_class}**")
    st.write(f"Confiança: **{confidence:.2%}**")
    st.write(f"Probabilidade da classe 1: `{probability:.4f}`")

st.divider()
st.subheader("Métricas do modelo")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
else:
    st.info("As métricas serão exibidas após o treinamento.")

if CONFUSION_MATRIX_PATH.exists():
    st.image(str(CONFUSION_MATRIX_PATH), caption="Matriz de Confusão")
