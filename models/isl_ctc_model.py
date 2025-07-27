import tensorflow as tf
from tensorflow.keras import layers, models

def build_isl_ctc_model(input_shape=(60, 224, 224, 3), vocab_size=50):
    """
    Builds a CNN + BiLSTM + CTC model for continuous sign recognition.
    """
    frames_input = layers.Input(shape=input_shape, name="frames_input")

    # TimeDistributed CNN (e.g., MobileNetV2)
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=input_shape[1:],
        include_top=False,
        weights="imagenet",
        pooling='avg'
    )
    base_cnn.trainable = False  # optionally freeze
    cnn_out = layers.TimeDistributed(base_cnn)(frames_input)

    # BiLSTM sequence model
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(cnn_out)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Output logits for each time step
    logits = layers.Dense(vocab_size + 1, activation='linear', name="logits")(x)  # +1 for CTC blank

    model = models.Model(inputs=frames_input, outputs=logits)
    return model
