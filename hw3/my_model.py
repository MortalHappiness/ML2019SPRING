# build model(cnn)

import numpy as np

from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.utils import plot_model

# ======================================

def build_model():
	model = Sequential()
	model.add(Conv2D(64,
					 input_shape = (48, 48, 1),
					 kernel_size = (5, 5),
					 padding = 'same',
					 kernel_initializer = 'glorot_normal'))
	model.add(LeakyReLU(alpha = 0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
	model.add(Dropout(0.25))

	model.add(Conv2D(128,
					 kernel_size = (3, 3),
					 padding = 'same',
					 kernel_initializer = 'glorot_normal'))

	model.add(LeakyReLU(alpha = 0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(256,
					 kernel_size = (3, 3),
					 padding = 'same',
					 kernel_initializer = 'glorot_normal'))

	model.add(LeakyReLU(alpha = 0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
	model.add(Dropout(0.35))

	model.add(Conv2D(512,
					 kernel_size = (3, 3),
					 padding = 'same',
					 kernel_initializer = 'glorot_normal'))

	model.add(LeakyReLU(alpha = 0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
	model.add(Dropout(0.4))

	model.add(Flatten())

	model.add(Dense(512, activation = 'relu', kernel_initializer = 'glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(512, activation = 'relu', kernel_initializer = 'glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(7, activation = 'softmax', kernel_initializer = 'glorot_normal'))

	return model

# =======================================

if __name__ == '__main__':
	model = build_model()
	model.summary()
	# plot_model(model,
	# 		   to_file = './hw3_model/model.png',
	# 		   show_shapes = True,
	# 		   show_layer_names = False)