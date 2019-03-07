# training, store the model as "./hw1_model/hw1_model.npy"
# type "python hw1_train.py" to execute

# =========================================

import numpy as np
import os

# =========================================

def save_mean_std(x_train):
	'''
	Save the mean and std for x_train
	'''
	x_mean = np.mean(x_train, axis = 0)
	x_std = np.std(x_train, axis = 0)
	np.save('./model/x_mean.npy', x_mean)
	np.save('./model/x_std.npy', x_std)

def feature_scaling(x_data):
	'''
	Returns: numpy array, same shape as x_data
	'''
	x_mean = np.load('./model/x_mean.npy')
	x_std = np.load('./model/x_std.npy')
	return (x_data - x_mean)/x_std

def training(x_train, y_train):
	'''
	Returns:
		w: A (163,) numpy array.
	'''
	# add bias
	x_train = np.concatenate((np.ones((5652, 1)), x_train), axis = 1)
	# declare the weight vector
	if os.path.exists('./model/hw1_model.npy'):
		print('hw1_model.npy has exist, loading the model...')
		w = np.load('./model/hw1_model.npy')
	else:
		w = np.zeros(163)
	# declare a vector to store the sum of the squares of the past gradients for adagrad
	sum_grad = np.zeros(163)
	sum_grad += 1e-8 # avoid division by zero
	# declare the initial learning rate and number of iteration
	l_rate = 0.01
	repeat = 100000
	# training
	for i in range(repeat):
		_y = np.dot(x_train, w)
		loss = _y - y_train
		print('iteration %d, error: %.3f' %(i, np.sqrt(np.sum(loss**2)/loss.shape[0])))
		grad = np.dot(x_train.T, loss)
		sum_grad += grad**2
		adagrad = np.sqrt(sum_grad)
		w -= l_rate*grad/adagrad
	return w

def main():
	x_train = np.load('./model/x_train.npy')
	y_train = np.load('./model/y_train.npy')

	# # feature scaling
	# save_mean_std(x_train)
	# x_train = feature_scaling(x_train)

	w = training(x_train, y_train)
	# save the model
	np.save('./hw1_model/hw1_model.npy', w)

# =========================================

if __name__ == '__main__':
	main()