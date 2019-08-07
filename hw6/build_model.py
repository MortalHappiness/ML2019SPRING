# build the RNN

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, MaxoutDense, Dropout, BatchNormalization, Flatten, \
                         Embedding, LSTM, GRU, \
                         Bidirectional, TimeDistributed
from keras.initializers import Constant
from keras import backend as K
from keras.engine import InputSpec
from keras.optimizers import Adam

from gensim.models import KeyedVectors

import preprocess

# ===================================
def get_embedding_layer():
    tokenizer = preprocess.create_tokenizer(None, './hw6_model/tokenizer')
    word_index = tokenizer.word_index
    wv = KeyedVectors.load('./hw6_model/embedding.kv')
    v = wv.vectors
    num_words, embedding_dim = v.shape

    num_words += 2 # preserve a space for 0(padding) and <oov>
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if word != '<oov>':
            embedding_vector = wv[word]
        else:
            embedding_vector = np.random.uniform(0, 1, embedding_dim)
        embedding_matrix[i] = embedding_vector

    max_len = preprocess.get_max_len()

    embedding_layer = Embedding(input_dim = num_words,
                                output_dim = embedding_dim,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = max_len,
                                trainable = True)

    return embedding_layer

def swish(x):
    return (K.sigmoid(x) * x)

# ref.: https://github.com/keras-team/keras/issues/7290
class TimestepDropout(Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape

def rnn_model(weights_path = None):
    model = Sequential()
    embedding_layer = get_embedding_layer()
    model.add(embedding_layer)
    model.add(TimestepDropout(0.5))
    model.add(Bidirectional(LSTM(128,
                                 dropout = 0.2,
                                 recurrent_dropout = 0.2,
                                 kernel_initializer = 'he_uniform',
                                 return_sequences = True)))
    model.add(Bidirectional(LSTM(64,
                                 dropout = 0.35,
                                 recurrent_dropout = 0.3,
                                 kernel_initializer = 'he_uniform',
                                 return_sequences = True)))
    model.add(TimeDistributed(Dense(8, activation = swish)))
    # model.add(TimeDistributed(MaxoutDense(8)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation = swish))
    # model.add(MaxoutDense(128))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation = swish))
    # model.add(MaxoutDense(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))

    if weights_path != None:
        model.load_weights(weights_path)

    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 0.0005),
                  metrics = ['accuracy'])

    return model

# ===================================

if __name__ == '__main__':
    model = rnn_model()
    model.summary()