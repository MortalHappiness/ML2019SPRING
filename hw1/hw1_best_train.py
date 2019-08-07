# training, store the model as "./hw1_best_model/hw1_best_model.h5"
# type "python hw1_best_train.py" to execute

# =========================================

import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.utils import resample

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Lambda
from keras import backend as K

from my_linear_regression import LinearRegression

# =========================================

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

def calculate_rmse(y, y_hat):
    '''
    Returns: rmse
    '''
    return np.sqrt(np.mean((y - y_hat)**2))

def reshape_to_2D(tensor):
    '''
    Returns: Reshaped tensor
    '''
    return K.reshape(tensor, (-1, 18, 9, 1))

# ===========================================
# models

def support_vector(x_train_m, y_train_m):
    model = SVR(kernel = 'linear', verbose = True)
    model.fit(x_train_m, y_train_m)
    return model

def random_forest(x_train_m, y_train_m):
    model = RandomForestRegressor(
                n_estimators = 100,
                bootstrap = True,
                oob_score = True,
                warm_start = True,
                verbose = 1,
                n_jobs = -1
                )
    model.fit(x_train_m, y_train_m)
    return model

def extra_trees(x_train_m, y_train_m):
    model = ExtraTreesRegressor(
                n_estimators = 100,
                bootstrap = True,
                oob_score = True,
                warm_start = True,
                verbose = 1,
                n_jobs = -1
                )
    model.fit(x_train_m, y_train_m)
    return model

def cnn(x_train_m, y_train_m, batch_split = 1):
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

    # train model
    model.fit(x_train_m, y_train_m,
              batch_size = len(x_train_m)//batch_split,
              epochs = 370//batch_split + 130,
              shuffle = True,
              )
    return model

def linear_regression(x_train_m, y_train_m):
    model = LinearRegression(repeat = 50000)
    model.fit(x_train_m, y_train_m)

    return model

# =======================================

def train_models(x_train_m, y_train_m):
    '''
    Returns: list of model
    '''
    models = list()

    model_1 = support_vector(x_train_m, y_train_m)
    models.append(model_1)
    joblib.dump(model_1, './hw1_best_model/models/model_1.pkl')

    model_2 = random_forest(x_train_m, y_train_m)
    models.append(model_2)
    joblib.dump(model_2, './hw1_best_model/models/model_2.pkl')

    model_3 = extra_trees(x_train_m, y_train_m)
    models.append(model_3)
    joblib.dump(model_3, './hw1_best_model/models/model_3.pkl')

    model_4 = cnn(x_train_m, y_train_m)
    models.append(model_4)
    model_4.save('./hw1_best_model/models/model_4.h5')

    model_5 = linear_regression(x_train_m, y_train_m)
    models.append(model_5)
    model_5.save('./hw1_best_model/models/model_5.npy')

    return models

def stacking(models, x_train_c, y_train_c):
    '''
    Returns: A model.
    '''
    # let every model predict the result
    results = list()
    for model in models:
        result = model.predict(x_train_c)
        if result.ndim == 2:
            result = result[:, 0]
        results.append(result)
    results = np.array(results).T

    classifier = support_vector(results, y_train_c)

    print(results)
    print(y_train_c)

    # save the classifier
    joblib.dump(classifier, './hw1_best_model/models/classifier.pkl')

    return classifier

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

def training(x_data, y_data):
    '''
    Returns: models, classifier
    '''
    # split training and validation data
        # x_train_m, y_train_m: used for training models
        # x_train_c, y_train_c: used for training classifier
        # x_test, y_test: the final validation data
    x_train, x_test, y_train, y_test = \
            train_test_split(x_data, y_data, test_size = 1)
    x_train_m, x_train_c, y_train_m, y_train_c = \
            train_test_split(x_train, y_train, test_size = 0.4)
    # feature scaling
    x_mean, x_std = calculate_mean_std(x_train_m)
    np.save('./hw1_best_model/x_mean.npy', x_mean)
    np.save('./hw1_best_model/x_std.npy', x_std)
    x_train_m = feature_scaling(x_train_m, x_mean, x_std)
    x_train_c = feature_scaling(x_train_c, x_mean, x_std)
    x_test = feature_scaling(x_test, x_mean, x_std)
    # train models
    models = train_models(x_train_m, y_train_m)
    # train the classifier
    classifier = stacking(models, x_train_c, y_train_c)
    # evaluate the model
    predicted = predict_result(classifier, models, x_test)
    print('='*20)
    print('Test rmse: %.3f' 
            %(calculate_rmse(predicted, y_test)))

    return models, classifier

def main():
    x_data = np.load('./model/x_train.npy')
    y_data = np.load('./model/y_train.npy')

    # training
    models, classifier = training(x_data, y_data)

    # save the model
    # joblib.dump(model, './model/model.pkl')

# =========================================

if __name__ == '__main__':
    main()