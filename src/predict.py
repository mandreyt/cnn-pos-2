import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

from config import MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE


def load_class_names():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file:
        class_indices = json.load(file)

    # Inverte o dicionário: {"cat": 0, "dog": 1} -> {0: "cat", 1: "dog"}
    return {value: key for key, value in class_indices.items()}


def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    image = load_img(image_path, target_size=IMAGE_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array)[0][0])
    predicted_class_index = 1 if probability > 0.5 else 0
    predicted_class = class_names[predicted_class_index]

    confidence = probability if predicted_class_index == 1 else 1 - probability

    return predicted_class, confidence, probability


def main():
    parser = argparse.ArgumentParser(description="Predição de imagem usando CNN treinada.")
    parser.add_argument("--image", required=True, help="Caminho da imagem para classificar.")
    args = parser.parse_args()

    predicted_class, confidence, probability = predict_image(args.image)

    print("Classe prevista:", predicted_class)
    print(f"Confiança: {confidence:.2%}")
    print(f"Probabilidade da classe 1: {probability:.4f}")


if __name__ == "__main__":
    main()
