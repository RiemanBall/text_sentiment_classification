# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Dense, LayerNormalization
from kerastuner.engine.hypermodel import HyperModel

class BiLstmTuner(HyperModel):
    def __init__(self, embedding, train_embedding, sen_len):
        vocab_dim, embed_dim = embedding.shape
        self.embedding_layer = Embedding(vocab_dim, embed_dim, trainable = train_embedding) 
        self.embedding_layer.build((None,))
        self.embedding_layer.set_weights([embedding])

        self.sen_len = sen_len

    def build(self, hp):
        sentence_indices = Input(shape = (self.sen_len,), dtype = 'int32')
        dp_rate = hp.Float('dp', min_value = 0.5, max_value = 0.8)

        x = self.embedding_layer(sentence_indices)
        x = Dropout(dp_rate)(x)
        x = Bidirectional(LSTM(units = hp.Int('lstm1', min_value = 64, max_value = 128, step = 32), 
                               return_sequences = True, 
                               dropout = dp_rate ))(x)
        x = LayerNormalization(axis = 1)(x)
        x = Bidirectional(LSTM(units = hp.Int('lstm2', min_value = 64, max_value = 128, step = 32), 
                               dropout = dp_rate ))(x)
        x = LayerNormalization(axis = 1)(x)
        outputs = Dense(1)(x)

        model = tf.keras.Model(inputs = sentence_indices, outputs = outputs)

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Float('lr', 1e-6, 1e-3)),
            loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
            metrics = ['accuracy'],
        )

        return model



def buildModel(embedding, train_embedding, sen_len, hidden_dim1, hidden_dim2, dp_rate, lr):
    
    vocab_dim, embed_dim = embedding.shape
    embedding_layer = Embedding(vocab_dim, embed_dim, trainable = train_embedding) 
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding])

    sentence_indices = Input(shape = (sen_len,), dtype = 'int32')
    x = embedding_layer(sentence_indices)
    x = Dropout(dp_rate)(x)
    x = Bidirectional(LSTM(hidden_dim1, return_sequences = True, dropout = dp_rate))(x)
    x = LayerNormalization(axis = 1)(x)
    x = Bidirectional(LSTM(hidden_dim2, dropout = dp_rate))(x)
    x = LayerNormalization(axis = 1)(x)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs = sentence_indices, outputs = outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits = True),
        metrics=['accuracy'],
    )

    return model


def testing(model, dataset):
    outputs = model.predict(dataset)

    outputs_prob = tf.math.sigmoid(outputs.reshape(-1))

    res = np.array([1 if prob > 0.5 else 0 for prob in outputs_prob])

    return res