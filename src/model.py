import tensorflow as tf


def build_cnn(input_shape=(64, 64, 3)) -> tf.keras.Model:
    """Cria uma CNN simples para classificação binária de imagens."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(units=1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
