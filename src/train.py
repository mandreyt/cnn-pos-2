import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_DIR,
    REPORT_DIR,
    MODEL_PATH,
    CLASS_INDICES_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    EPOCHS
)
from data import create_train_generator, create_test_generator
from model import build_cnn


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    training_set = create_train_generator(TRAIN_DIR)
    test_set = create_test_generator(TEST_DIR)

    print("Classes encontradas:", training_set.class_indices)

    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as file:
        json.dump(training_set.class_indices, file, indent=4, ensure_ascii=False)

    print("Construindo modelo CNN...")
    cnn = build_cnn()
    cnn.summary()

    print("Iniciando treinamento...")
    cnn.fit(
        x=training_set,
        validation_data=test_set,
        epochs=EPOCHS
    )

    print("Salvando modelo...")
    cnn.save(MODEL_PATH)

    print("Calculando métricas no conjunto de teste...")
    probabilities = cnn.predict(test_set)
    y_pred = (probabilities > 0.5).astype(int).reshape(-1)
    y_true = test_set.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(test_set.class_indices.keys()),
            zero_division=0,
            output_dict=True
        )
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    display_labels = list(test_set.class_indices.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão - Dados de Teste")
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close()

    print("\n===== MÉTRICAS NO TESTE =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    print(f"\nModelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")


if __name__ == "__main__":
    main()
