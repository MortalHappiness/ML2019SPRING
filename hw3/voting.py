# voting test
# type "python voting.py {test.csv} {result.csv}" to execute

import sys
import csv
import time

import numpy as np
import pandas as pd

import all_models.model1
import all_models.model2
import all_models.model3
import all_models.model4
import all_models.model5

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
    x_test = np.array(x_test).astype('float32')/255
    x_test = x_test[:, :, :, np.newaxis]

    print('testing data loading finished')

    return x_test

def predict_result(model, x_test):
    result = model.predict(x_test)
    result = np.argmax(result, axis = 1)

    return result

def output_result(result, output_path):
    with open(output_path, 'w', newline = '') as fout:
        output = csv.writer(fout, delimiter = ',', lineterminator = '\n')
        output.writerow(['id', 'label'])
        ids = np.arange(len(result))[:, np.newaxis]
        result = result[:, np.newaxis]
        rows = np.concatenate((ids, result), axis = 1)
        output.writerows(rows)

def main(script, test_data, output_path):
    x_test = load_test_data(test_data)

    models = list()
    model1 = all_models.model1.build_model()
    model1.load_weights('./all_models/model1.h5')
    models.append(model1)

    model2 = all_models.model2.build_model()
    model2.load_weights('./all_models/model2.h5')
    models.append(model2)

    model3 = all_models.model3.build_model()
    model3.load_weights('./all_models/model3.h5')
    models.append(model3)

    model4 = all_models.model4.build_model()
    model4.load_weights('./all_models/model4.h5')
    models.append(model4)

    model5 = all_models.model5.build_model()
    model5.load_weights('./all_models/model5.h5')
    models.append(model5)

    predicts = list()

    for model in models:
        predicts.append(model.predict(x_test))

    # vote
    result = np.argmax(sum(predicts), axis = 1)

    output_result(result, output_path)

# ====================================

if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t
    print('Testing time: %.3f seconds.'%t)