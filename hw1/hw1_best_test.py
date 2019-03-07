# testing, store the result as the specified file name
# type "python hw1_best_test.py {outputfile}" to execute

# =============================================

import sys
import time
import csv
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Lambda
from keras import backend as K

from my_linear_regression import LinearRegression

# =========================================
def reshape_to_2D(tensor):
	'''
	Returns: Reshaped tensor
	'''
	return K.reshape(tensor, (-1, 18, 9, 1))
	
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

def feature_scaling(x_data, x_mean, x_std):
	'''
	Returns: numpy array, same shape as x_data
	'''
	return (x_data - x_mean)/x_std

def predict_result(classifier, models, x_data):
	'''
	Input:
		classifier: the stacking model(classifier)
		models: every model used in stacking
	'''
	# let every model predict the result
	results = list()
	for model in models:
		result = model.predict(x_data)
		if result.ndim == 2:
			result = result[:, 0]
		results.append(result)
	results = np.array(results).T
	# predict the result
	return classifier.predict(results)

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
	# load models
	models = list()
	model_1 = joblib.load('./hw1_best_model/models/model_1.pkl')
	models.append(model_1)
	model_2 = joblib.load('./hw1_best_model/models/model_2.pkl')
	models.append(model_2)
	model_3 = joblib.load('./hw1_best_model/models/model_3.pkl')
	models.append(model_3)
	model_4 = my_cnn_model()
	model_4.load_weights('./hw1_best_model/models/model4_weight.h5')
	models.append(model_4)
	model_5 = LinearRegression(w = np.load('./hw1_best_model/models/model_5.npy'))
	models.append(model_5)

	# load classifier
	classifier = joblib.load('./hw1_best_model/models/classifier.pkl')

	# load testing data
	x_test = np.load('./hw1_best_model/x_test.npy')

	# feature scaling
	x_mean = np.load('./hw1_best_model/x_mean.npy')
	x_std = np.load('./hw1_best_model/x_std.npy')
	x_test = feature_scaling(x_test, x_mean, x_std)

	# testing
	result = predict_result(classifier, models, x_test)

	# output the result
	output_result(result, output_file)

# =========================================

if __name__ == '__main__':
	t = time.perf_counter()
	main(*sys.argv)
	t = time.perf_counter() - t
	print('testing time: %.3f seconds' % t)