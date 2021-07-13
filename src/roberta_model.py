import tensorflow as tf


class RobertaModel:
    LEARNING_RATE = 1e-5
    MAX_LENGTH = 256
    EMBEDDING_DIM = 238

    def build(self, base_model):
        encoder = self._build_encoder(base_model)
        model = self._build_model(encoder)
        return model

    @classmethod
    def _build_encoder(cls, base_model):
        encoder = Encoder(embedding_dim=cls.EMBEDDING_DIM, base_model=base_model)
        return encoder

    @classmethod
    def _build_model(cls, encoder):
        input_ids = tf.keras.layers.Input(
            shape=(cls.MAX_LENGTH,), dtype=tf.int32, name="input_ids"
        )
        input_attention_mask = tf.keras.layers.Input(
            shape=(cls.MAX_LENGTH,), dtype=tf.int32, name="attention_mask"
        )

        outputs = encoder(input_ids=input_ids, attention_mask=input_attention_mask)

        model = tf.keras.Model(
            inputs=[input_ids, input_attention_mask], outputs=outputs
        )

        optimizer = tf.keras.optimizers.Adam(lr=cls.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        return model

class Encoder(tf.keras.Model):
    DROPOUT_RATE = 0.5

    def __init__(self, base_model, embedding_dim: int, output_dim: int = 1):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = tf.keras.layers.Dropout(self.DROPOUT_RATE)
        self.dense = tf.keras.layers.Dense(
            output_dim,
            input_shape=(embedding_dim,),
            activation=None,
            name="dense_layer",
        )
        self.bert_layer = base_model

    def call(self, input_ids, attention_mask):
        sequence = self.bert_layer(input_ids, attention_mask)[0]
        x = sequence[:, 0, :]
        x = self.dense(x)
        return x
