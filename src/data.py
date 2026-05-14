from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMAGE_SIZE, BATCH_SIZE, SEED


def create_train_generator(train_dir):
    """Cria gerador com data augmentation para os dados de treino."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    return train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=SEED
    )


def create_test_generator(test_dir):
    """Cria gerador dos dados de teste sem embaralhar, necessário para métricas corretas."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    return test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
