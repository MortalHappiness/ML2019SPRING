# testing, store the result as the specified file name
# type "python hw1_test.py {outputfile}" to execute

# =============================================

import sys
import csv
import numpy as np

# =========================================
def feature_scaling(x_data):
	'''
	Returns: numpy array, same shape as x_data
	'''
	x_mean = np.load('./hw1_model/x_mean.npy')
	x_std = np.load('./hw1_model/x_std.npy')
	return (x_data - x_mean)/x_std

def testing(x_test, w):
	'''
	Retruns: A (240,) numpy array.
	'''
	# add bias
	x_test = np.concatenate((np.ones((240, 1)), x_test), axis = 1)
	return np.dot(x_test, w)

def output_result(result, output_file):
	'''
	Write the result to output_file.
	'''
	with open(output_file, 'w', encoding = 'big5') as fout:
		op_rows = csv.writer(fout, delimiter = ',', lineterminator = '\n')
		op_rows.writerow(['id', 'value'])
		result_list = list()
		for i in range(240):
			result_list.append(['id_' + str(i), result[i]])
		op_rows.writerows(result_list)

def main(script, output_file):
	# load the pre-trained model
	w = np.load('./hw1_model/hw1_model.npy')
	x_test = np.load('./hw1_model/x_test.npy')

	# x_test = feature_scaling(x_test)
	
	result = testing(x_test, w)
	# output the result
	output_result(result, output_file)

# =========================================

if __name__ == '__main__':
	main(*sys.argv)