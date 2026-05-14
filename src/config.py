from pathlib import Path

# Diretórios principais do projeto
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "training_set"
TEST_DIR = DATA_DIR / "test_set"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Arquivos gerados
MODEL_PATH = MODEL_DIR / "cnn_model.keras"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"

# Hiperparâmetros simples para fins didáticos
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
