# best test
# type "python hw2_best_test.py {X_test} {hw2_best.csv}" to execute

import sys
import time
import csv
import warnings

import numpy as np
import pandas as pd

from sklearn.externals import joblib
# =============================
warnings.filterwarnings('ignore')

# =============================

def feature_scaling(x_data, x_mean, x_std):
    '''
    Returns: numpy array, same shape as x_data
    '''
    ans = (x_data - x_mean)/x_std
    # if x_std == 0, then (x_data - x_mean) == 0
    # simply set those invalid terms(nan and inf) to zero
    ans_nan = np.isnan(ans)
    ans_inf = np.isinf(ans)
    ans_invalid = np.logical_or(ans_nan, ans_inf)
    ans[ans_invalid] = 0
    return ans

def load_data(X_test):
    '''
    Returns: x_test (numpy array)
    '''
    df = pd.read_csv(X_test)
    x_test = df.to_numpy().astype('float64')

    return x_test

def convert_probability_to_classes(result):
    '''
    Returns: y (numpy array)
    '''
    mask = (result >= 0.5)
    y = np.zeros(len(result))
    y[mask] = 1

    return y.astype('int32')

def predict_result(model, x_test):
    result = model.predict(x_test)
    if result.ndim == 2:
        result = result.ravel()
    return convert_probability_to_classes(result)

def output_result(result, output):
    '''
    Write the result into output file
    '''
    with open(output, 'w') as fout:
        op_rows = csv.writer(fout, delimiter = ',', lineterminator = '\n')
        op_rows.writerow(['id', 'label'])
        result_list = list()
        for i in range(len(result)):
            result_list.append([i+1, result[i]])
        op_rows.writerows(result_list)

def main(script, X_test, output):
    # load model
    model = joblib.load('./hw2_best_model/model.pkl')

    # load testing data
    x_test = load_data(X_test)

    # feature scaling
    x_mean = np.load('./hw2_best_model/x_mean.npy')
    x_std = np.load('./hw2_best_model/x_std.npy')
    x_test = feature_scaling(x_test, x_mean, x_std)

    # testing
    result = predict_result(model, x_test)

    # output the result
    output_result(result, output)

# =============================

if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t
    print('Test time: %.3f seconds' %t)