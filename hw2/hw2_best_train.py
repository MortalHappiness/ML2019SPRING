# best train
# type "python hw2_best_train.py {X_train} {Y_train}" to execute

import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

# =============================
warnings.filterwarnings('ignore')

# =============================
def calculate_mean_std(x_train):
	'''
	Returns: mean and std
	'''
	x_mean = np.mean(x_train, axis = 0)
	x_std = np.std(x_train, axis = 0)
	return x_mean, x_std

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

def load_data(X_train, Y_train):
	'''
	Returns: x_data, y_data (both are numpy array)
	'''
	df_x = pd.read_csv(X_train)
	x_data = df_x.to_numpy().astype('float64')

	df_y = pd.read_csv(Y_train)
	y_data = df_y.to_numpy().astype('float64')

	return x_data, y_data

def accuracy(y, y_hat):
	'''
	Returns: The accuracy for y with respect to y_hat
	'''
	return (1 - np.sum(np.abs(y - y_hat))/len(y))

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
	return convert_probability_to_classes(result)

# =================================
# models
def gradient_boost(x_train, y_train, x_test, y_test):
	model = GradientBoostingClassifier(
				loss = 'deviance',
				learning_rate = 0.1,
				n_estimators = 1000,
				max_depth = 2,
				verbose = True)
	model.fit(x_train, y_train)

	# save model
	joblib.dump(model, './hw2_best_model/model.pkl')

	return model

# =================================
def train_model(x_train, y_train, x_test, y_test):
	'''
	Returns: the trained model
	'''
	model = gradient_boost(x_train, y_train, x_test, y_test)

	return model

def training(x_data, y_data):
	'''
	Returns: model
	'''
	# split training and validation data
		# x_train, y_train: used for training models
		# x_test, y_test: the validation data
	kf = KFold(3, shuffle = True)
	train_scores = list()
	test_scores = list()

	x_mean, x_std = calculate_mean_std(x_data)
	np.save('./hw2_best_model/x_mean.npy', x_mean)
	np.save('./hw2_best_model/x_std.npy', x_std)
	x_data = feature_scaling(x_data, x_mean, x_std)
	model = train_model(x_data, y_data, None, None)

	# for train_index, test_index in kf.split(x_data):
	# 	x_train, x_test = x_data[train_index], x_data[test_index]
	# 	y_train, y_test = y_data[train_index], y_data[test_index]
	# 	# feature scaling
	# 	x_mean, x_std = calculate_mean_std(x_train)
	# 	np.save('./hw2_best_model/x_mean.npy', x_mean)
	# 	np.save('./hw2_best_model/x_std.npy', x_std)
	# 	x_train = feature_scaling(x_train, x_mean, x_std)
	# 	x_test = feature_scaling(x_test, x_mean, x_std)
	# 	# train models
	# 	model = train_model(x_train, y_train, x_test, y_test)
	# 	# evaluate the model
	# 	train_predict = predict_result(model, x_train)
	# 	test_predict = predict_result(model, x_test)
	# 	train_scores.append(accuracy(train_predict, y_train))
	# 	test_scores.append(accuracy(test_predict, y_test))

	# 	break

	# train_scores = np.array(train_scores)
	# test_scores = np.array(test_scores)
	# print('Train score:', train_scores, ',avg = %.3f'%np.mean(train_scores))
	# print('Test score:', test_scores, ',avg = %.3f'%np.mean(test_scores))

	return model

def main(script, X_train, Y_train):
	x_data, y_data = load_data(X_train, Y_train)
	# flatten the y_data
	y_data = y_data.ravel()
	# training
	model = training(x_data, y_data)

# =============================

if __name__ == '__main__':
	main(*sys.argv)