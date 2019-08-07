# functions for data loading

import csv

import numpy as np
from keras.utils import to_categorical

# =======================================

def load_train_data(train_data_path):
    '''
    Load the training data.

    # Arguments:
        train_data_path(str): The path to train.csv

    # Returns:
        x_train(array): shape = (28709, 48, 48, 1)
        y_train(array): shape = (28709, 7)
    '''
    print('loading the training data ...')

    x_train = list()
    y_train = list()
    with open(train_data_path, 'r', newline = '') as fin:
        rows = csv.reader(fin)
        next(rows) # skip the header
        for row in rows:
            label, feature = row
            feature = [int(x) for x in feature.split()]
            feature = np.array(feature).reshape(48, 48)
            x_train.append(feature)
            y_train.append(int(label))
    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train)

    x_train = x_train[:, :, :, np.newaxis]
    y_train = to_categorical(y_train)

    print('training data loading finished')

    return x_train, y_train

def load_test_data(test_data):
    '''
    Load the testing data.

    # Arguments:
        test_data(str): The path to test.csv

    # Returns:
        x_test(array): shape = (28709, 48, 48, 1)
    '''
    print('loading the testing data ...')

    x_test = list()
    with open(test_data, 'r', newline = '') as fin:
        rows = csv.reader(fin)
        next(rows) # skip the header
        for row in rows:
            _, feature = row
            feature = [int(x) for x in feature.split()]
            feature = np.array(feature).reshape(48, 48)
            x_test.append(feature)
    x_test = np.array(x_test).astype('float32')
    x_test = x_test[:, :, :, np.newaxis]

    print('testing data loading finished')

    return x_test