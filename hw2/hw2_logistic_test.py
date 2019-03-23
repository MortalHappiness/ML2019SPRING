# logistic test
# type "python hw2_logistic_test.py {X_test} {hw2_logistic.csv}" to execute

import sys
import time
import csv

import numpy as np
import pandas as pd

# ==============================
def feature_scaling(x_data, x_mean, x_std):
	'''
	Returns: numpy array, same shape as x_data
	'''
	return (x_data - x_mean)/x_std

def sigmoid(z):
	ans = 1.0/(1.0 + np.exp(-z))
	return np.clip(ans, 1e-15, 1 - 1e-15) # avoid overflow

def load_data(X_test):
	'''
	Returns: x_test (numpy array)
	'''
	df = pd.read_csv(X_test)
	x_test = df.to_numpy().astype('float64')

	return x_test

def testing(x_test, w):
	'''
	Returns: y (numpy array)
	'''
	# add bias
	x_test = np.concatenate(
				(np.ones((x_test.shape[0], 1)), x_test), axis = 1)
	# predict
	prob = sigmoid(np.dot(x_test, w)) # probability of being class 1
	mask = (prob >= 0.5)
	y = np.zeros(len(x_test))
	y[mask] = 1

	return y.astype('int32')

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
	x_test = load_data(X_test)

	# feature scaling
	x_mean = np.load('./hw2_logistic_model/x_mean.npy')
	x_std = np.load('./hw2_logistic_model/x_std.npy')
	x_test = feature_scaling(x_test, x_mean, x_std)

	w = np.load('./hw2_logistic_model/model.npy')
	result = testing(x_test, w)
	output_result(result, output)

# =============================

if __name__ == '__main__':
	main(*sys.argv)