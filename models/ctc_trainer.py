import tensorflow as tf

class CTCModel(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(CTCModel, self).__init__(**kwargs)
        self.base_model = base_model
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self.loss_fn(
                y["labels"],
                y_pred,
                y["input_length"],
                y["label_length"]
            )
        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        return {"loss": tf.reduce_mean(loss)}

    def test_step(self, data):
        x, y = data
        y_pred = self.base_model(x, training=False)
        loss = self.loss_fn(
            y["labels"],
            y_pred,
            y["input_length"],
            y["label_length"]
        )
        return {"val_loss": tf.reduce_mean(loss)}

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
