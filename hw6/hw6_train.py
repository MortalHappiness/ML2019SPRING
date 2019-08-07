# training
# type "python hw6_train.py {train_x.csv} {train_y.csv} {dict.txt.big}" to execute

import sys
import time
import os
import shutil
import random

import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

import preprocess
import build_model

# =======================================
def training(model, x_train, y_train, vali_data):
    checkpoint = ModelCheckpoint('./checkpoint/weights'
                                 '_{epoch:03d}'
                                 '_{loss:.2f}'
                                 '_{val_loss:.2f}'
                                 '_{acc:.2f}'
                                 '_{val_acc:.2f}.h5',
                                 save_weights_only = True,
                                 period = 1)

    if os.path.exists('./checkpoint'):
        shutil.rmtree('./checkpoint')
    os.mkdir('./checkpoint')

    model.fit(x_train, y_train,
              validation_data = vali_data,
              batch_size = 2048,
              epochs = 20,
              callbacks = [checkpoint])

    model.save_weights('./hw6_model/model_weights.h5')

def main(script, train_x, train_y, dict_txt):
    # load data
    print('loading data and tokenizer...')
    x_data = preprocess.load_train(train_x)
    y_data = preprocess.load_label(train_y)
    tokenizer = preprocess.create_tokenizer(None, './hw6_model/tokenizer')
    wv = KeyedVectors.load('./hw6_model/embedding.kv')

    print('tokenize and pad the data...')
    x_data = preprocess.tokenize(x_data, dict_txt)
    x_data = preprocess.replace_with_oov(x_data, wv)
    x_data = tokenizer.texts_to_sequences(x_data)
    max_len = preprocess.get_max_len()

    x_data = pad_sequences(x_data, maxlen = max_len)
    print('tokenize and padding finished!')

    # print('spliting training and testing data...')
    print('sampling training data(bagging)...')
    sample_idx = np.random.randint(len(x_data), size = (300000,))
    remaining_idx = np.array(list(set(np.arange(len(x_data))) - set(sample_idx)))
    train_idx = sample_idx
    test_idx = remaining_idx
    # train_idx = np.random.permutation(np.arange(len(x_data))) < 130000
    # test_idx = np.logical_not(train_idx)
    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_test = x_data[test_idx]
    y_test = y_data[test_idx]
    vali_data = (x_test, y_test)

    # training
    model = build_model.rnn_model()
    training(model, x_train, y_train, vali_data)

# =======================================

if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t
    print('Executing time: %.3f seconds.' % t)
