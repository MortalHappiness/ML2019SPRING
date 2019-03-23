# generative train
# type "python hw2_generative_train.py {X_train} {Y_train}" to execute

import sys

import numpy as np
import pandas as pd

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
	return (x_data - x_mean)/x_std

def load_data(X_train, Y_train):
	'''
	Returns: x_data, y_data (both are numpy array)
	'''
	df_x = pd.read_csv(X_train)
	x_data = df_x.to_numpy().astype('float64')

	df_y = pd.read_csv(Y_train)
	y_data = df_y.to_numpy().astype('float64')

	return x_data, y_data

def sigmoid(z):
	ans = 1.0/(1.0 + np.exp(-z))
	return np.clip(ans, 1e-15, 1 - 1e-15) # avoid overflow

def generative(x_train, y_train):
	'''
	Returns: w (the weight vector)
	'''
	# flatten y_train
	y_train = y_train.ravel()

	# split x_train into two parts: class 0 and class 1
	mask0 = (y_train < 0.5) # 0
	mask1 = (y_train > 0.5) # 1
	x_train_0 = x_train[mask0] # select all data belonging to class 0
	x_train_1 = x_train[mask1] # select all data belonging to class 1

	# calculate mu0, mu1, N0, N1
		# mu0, mu1: mean of the guassian distributions
		#			to sample class 0 and 1
		# N0, N1: number of class 0 and class 1 in the training data
	N0 = len(x_train_0)
	N1 = len(x_train_1)
	mu0 = np.mean(x_train_0, axis = 0)
	mu1 = np.mean(x_train_1, axis = 0)

	# calculate sigma0, sigma1, share_sigma
		# sigma0, sigma1: covariance of the guassian distributions
		#				  to sample class 0 and class 1
		# share_sigma: the shared_sigma between class 0 and class 1
	sigma0 = np.zeros((x_train.shape[1], x_train.shape[1]))
	front0 = (x_train_0 - mu0)[:, :, np.newaxis] # the front part of dot product
	back0 = (x_train_0 - mu0)[:, np.newaxis, :] # the back part of dot product
	for i in range(N0):
		sigma0 += np.dot(front0[i], back0[i])
	sigma0 /= N0

	sigma1 = np.zeros((x_train.shape[1], x_train.shape[1]))
	front1 = (x_train_1 - mu1)[:, :, np.newaxis] # the front part of dot product
	back1 = (x_train_1 - mu1)[:, np.newaxis, :] # the back part of dot product
	for i in range(N1):
		sigma1 += np.dot(front1[i], back1[i])
	sigma1 /= N1

	share_sigma = (N0*sigma0 + N1*sigma1)/(N0 + N1)

	# calculate w
	sigma_inverse = np.linalg.inv(share_sigma)
	w = np.dot((mu0 - mu1), sigma_inverse)
	b = (-0.5)*np.dot(np.dot(mu0, sigma_inverse), mu0) \
		+ (0.5)*np.dot(np.dot(mu1, sigma_inverse), mu1) \
		+ np.log(float(N0)/N1)
	b = b.reshape(1)
	w = np.concatenate((b, w), axis = 0)

	return w

def main(script, X_train, Y_train):
	x_data, y_data = load_data(X_train, Y_train)

	# feature scaling
	x_mean, x_std = calculate_mean_std(x_data)
	x_data = feature_scaling(x_data, x_mean, x_std)
	np.save('./hw2_generative_model/x_mean.npy', x_mean)
	np.save('./hw2_generative_model/x_std.npy', x_std)

	w = generative(x_data, y_data)
	np.save('./hw2_generative_model/model.npy', w)

# =============================

if __name__ == '__main__':
	main(*sys.argv)