# testing
# type "python hw8_test.py {test.csv} {result.csv}" to execute

import sys
import csv
import time

import numpy as np

import load_data
from my_model import CompressedModel

# ====================================

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

def output_result(result, output_path):
    with open(output_path, 'w', newline = '') as fout:
        output = csv.writer(fout, delimiter = ',', lineterminator = '\n')
        output.writerow(['id', 'label'])
        ids = np.arange(len(result))[:, np.newaxis]
        result = result[:, np.newaxis]
        rows = np.concatenate((ids, result), axis = 1)
        output.writerows(rows)

def main(script, test_data, output_path):
    x_test = load_data.load_test_data(test_data)

    model = CompressedModel()
    model.load_weights()

    result = model.testing(x_test)

    output_result(result, output_path)

# ====================================

if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t
    print('Testing time: %.3f seconds.' % t)
