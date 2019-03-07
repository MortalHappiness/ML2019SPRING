# read "test.csv" and store the processed data as "x_test.npy"
# type "python process_test_data.py {inputfile} {output}" to execute

# =============================================

import sys
import csv
import numpy as np

# =========================================

def my_cnn_model():
	# create model
	input_1D = Input(shape = (162,))
	input_2D = Lambda(reshape_to_2D)(input_1D)
	conv = Conv2D(filters = 40,
				  kernel_size = (18, 3)
				  )(input_2D)
	flat = Flatten()(conv)
	dense = Dense(units = 400,
				  activation = 'relu'
				  )(flat)
	drop = Dropout(0.63)(dense)
	output = Dense(units = 1,
				   activation = 'relu'
				   )(drop)
	model = Model(inputs = input_1D, outputs = output)

	model.compile(loss = 'mse',
				  optimizer = 'adam',
				  metrics = ['mse'])

	return model

def load_test_data(test_file):
	'''
	Returns: A (240, 162) numpy array.
	'''
	x_test = list()
	with open(test_file, 'r', newline = '', encoding = 'big5') as fin:
		rows = csv.reader(fin, delimiter = ',')
		n_row = 0
		for row in rows:
			if n_row%18 == 0:
				x_test.append(list())
			for i in range(9):
				if row[i+2] == 'NR':
					x_test[n_row//18].append(float(0))
				else:
					x_test[n_row//18].append(float(row[i+2]))
			n_row += 1
	return np.array(x_test)

def main(script, test_file, output):
	x_test = load_test_data(test_file)
	np.save(output, x_test)

# =========================================

if __name__ == '__main__':
	main(*sys.argv)