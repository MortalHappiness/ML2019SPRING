import numpy as np

class LinearRegression:
    def __init__(self, repeat = None, w = None):
        self.repeat = repeat
        self.w = w

    def fit(self, x_train, y_train):
        # add bias
        x_train = np.concatenate((np.ones((len(x_train), 1)), x_train), axis = 1)
        # declare the weight vector
        w = np.zeros(163)
        # declare a vector to store the sum of the squares of the past gradients for adagrad
        sum_grad = np.zeros(163)
        sum_grad += 1e-8 # avoid division by zero
        # declare the initial learning rate and number of iteration
        l_rate = 10
        # training
        for i in range(self.repeat):
            _y = np.dot(x_train, w)
            loss = _y - y_train
            print('iteration %d, error: %.3f' %(i, np.sqrt(np.sum(loss**2)/loss.shape[0])))
            grad = np.dot(x_train.T, loss)
            sum_grad += grad**2
            adagrad = np.sqrt(sum_grad)
            w -= l_rate*grad/adagrad
        self.w = w

    def save(self, path):
        np.save(path, self.w)

    def predict(self, x_test):
        x_test = np.concatenate((np.ones((len(x_test), 1)), x_test), axis = 1)
        return np.dot(x_test, self.w)