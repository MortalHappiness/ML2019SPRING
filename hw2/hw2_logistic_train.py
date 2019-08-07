# logistic train
# type "python hw2_logistic_train.py {X_train} {Y_train}" to execute

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

def cross_entropy(y, y_hat):
    '''
    Calulate the cross entropy between y and y_hat
    '''
    cross_entropy = -(np.dot(y_hat, np.log(y)) \
                    + np.dot((1 - y_hat), np.log(1 - y)))
    return np.sum(cross_entropy)

def training(x_train, y_train):
    '''
    Returns: w (the weight vector)
    '''
    # add bias
    x_train = np.concatenate(
                (np.ones((x_train.shape[0], 1)), x_train), axis = 1)
    # declare hyperparameters
    w = np.zeros(x_train.shape[1])
    learn_rate = 1
    repeat = 10000
    sum_grad = np.zeros(w.shape[0])
    sum_grad += 1e-15 # avoid division by zero
    # flatten the y_train
    y_train = y_train.ravel()
    # logistic regression
    for i in range(repeat):
        y = sigmoid(np.dot(x_train, w))
        loss = y - y_train
        grad = np.dot(x_train.T, loss)
        sum_grad += grad**2
        adagrad = np.sqrt(sum_grad)
        w -= learn_rate*grad/adagrad
        if (i % 100 == 0):
            print('Iteration %d | Loss(cross_entropy) = %.3f'
                    %(i, cross_entropy(y, y_train)))
    return w

def main(script, X_train, Y_train):
    x_data, y_data = load_data(X_train, Y_train)

    # feature scaling
    x_mean, x_std = calculate_mean_std(x_data)
    x_data = feature_scaling(x_data, x_mean, x_std)
    np.save('./hw2_logistic_model/x_mean.npy', x_mean)
    np.save('./hw2_logistic_model/x_std.npy', x_std)

    w = training(x_data, y_data)
    np.save('./hw2_logistic_model/model.npy', w)

# =============================

if __name__ == '__main__':
    main(*sys.argv)