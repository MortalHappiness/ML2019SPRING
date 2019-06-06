# the model (network compression)

import os

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, \
						 Activation, ReLU, Conv2D, DepthwiseConv2D, \
						 GlobalAveragePooling2D, Add
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# ======================================
# blocks

def inverted_res_block(input_tensor, filters, stride, expand, expansion):
	'''
	# Arguments:
		input_tensor(input_tensor)
		filters(int): the filters in the last pointwise conv layer
		stride(int): the stride in the middle depthwise conv layer
		expand(bool): whether to use one conv layer to expand channels
		expansion(int): the expansion factor in first conv layer

	# Reference:
		https://arxiv.org/pdf/1801.04381.pdf
	'''
	input_filters = K.int_shape(input_tensor)[-1]
	x = input_tensor

	# Expand
	if expand:
		x = Conv2D(expansion * input_filters,
				   kernel_size = 1,
				   padding = 'same',
				   use_bias = False,
				   activation = None)(x)
		x = BatchNormalization(momentum = 0.999)(x)
		x = ReLU(6.0)(x)

	# Depthwise
	if stride == 1:
		x = DepthwiseConv2D(kernel_size = 3,
						    strides = 1,
						    activation = None,
						    use_bias = False,
						    padding = 'same')(x)
	elif stride == 2:
		x = DepthwiseConv2D(kernel_size = 3,
							strides = 2,
							activation = None,
							use_bias = False,
							padding = 'same')(x)

	x = BatchNormalization(momentum = 0.999)(x)
	x = ReLU(6.0)(x)

	# Project
	x = Conv2D(filters,
			   kernel_size = 1,
			   padding = 'same',
			   use_bias = False,
			   activation = None)(x)
	x = BatchNormalization(momentum = 0.999)(x)

	if input_filters == filters and stride == 1:
		return Add()([input_tensor, x])

	return x

# ======================================

class CompressedModel:
	'''
	The model using network compression techniques.

	# Attributes:
		model(keras model)
		batch_size(int)
		epochs(int)
		weights_folder(str): the folder for saving the training weights
	'''
	def __init__(self, batch_size = 512, epochs = 100, weights_folder = './model'):
		self.batch_size = batch_size
		self.epochs = epochs
		self.weights_folder = weights_folder
		self.build_model()

	def build_model(self):
		input_tensor = Input(shape = (48, 48, 1))

		x = input_tensor

		x = Conv2D(32,
				   kernel_size = 3,
				   strides = 2,
				   padding = 'same',
				   use_bias = False)(x)
		x = BatchNormalization(momentum = 0.999)(x)
		x = ReLU(6.0)(x)

		x = inverted_res_block(x, filters = 25, stride = 1, expand = False, expansion = 1)

		x = inverted_res_block(x, filters = 30, stride = 2, expand = True, expansion = 4)
		x = inverted_res_block(x, filters = 30, stride = 1, expand = True, expansion = 4)

		x = inverted_res_block(x, filters = 60, stride = 2, expand = True, expansion = 4)
		x = inverted_res_block(x, filters = 60, stride = 1, expand = True, expansion = 4)
		x = inverted_res_block(x, filters = 60, stride = 1, expand = True, expansion = 4)

		x = Conv2D(70,
				   kernel_size = 1,
				   use_bias = False)(x)
		x = BatchNormalization(momentum = 0.999)(x)
		x = ReLU(6.0)(x)

		x = GlobalAveragePooling2D()(x)

		x = Dense(7, activation = 'softmax')(x)

		model = Model(input_tensor, x)

		self.model = model

	def load_weights(self):
		weights_npz = np.load(os.path.join(self.weights_folder, 'model.npz'))
		weights_files = weights_npz.files
		weights = [weights_npz['weight%d' % i] for i in range(len(weights_files))]

		# convert weights to float32
		weights = [weight.astype('float32') for weight in weights]

		self.model.set_weights(weights)

	def save_weights(self):
		weights = self.model.get_weights()

		# convert weights to float16
		weights = [('weight%d' % i, weight.astype('float16')) for i, weight in enumerate(weights)]
		weights = dict(weights)

		np.savez_compressed(os.path.join(self.weights_folder, 'model.npz'), **weights)

	def testing(self, x_test):
		result = self.model.predict(x_test, batch_size = self.batch_size)
		result = np.argmax(result, axis = 1)

		return result

	def distill(self, x_train, y_train, x_test, y_test):
		from teacher_model import Teacher

		self.model.compile(loss = 'categorical_crossentropy',
						   optimizer = Adam(),
						   metrics = ['accuracy'])

		if os.path.exists('./checkpoint/model.h5'):
			print('#'*40)
			print('Continuing from last training...')
			print('#'*40)
			self.model = load_model('./checkpoint/model.h5')

		generator = ImageDataGenerator(
						rotation_range = 25,
						width_shift_range = 0.2,
						height_shift_range = 0.2,
						shear_range = 0.2,
						zoom_range = [0.8, 1.2],
						horizontal_flip = True)
		generator.fit(x_train)
		generator = generator.flow(x_train, batch_size = self.batch_size)

		teacher = Teacher(generator,
						  batch_size = self.batch_size,
						  mean = x_train.mean(),
						  std = x_train.std())

		self.model.fit_generator(
					teacher,
					workers = 0,
					steps_per_epoch = len(x_train)//self.batch_size,
					epochs = self.epochs,
					validation_data = (x_test, y_test))
		self.save_weights()

		# save whole model for training
		self.model.save('./checkpoint/model.h5')

# =======================================

if __name__ == '__main__':
	from keras.utils import plot_model
	
	model = CompressedModel().model
	model.summary()
	plot_model(model,
			   to_file = './model/model.png',
			   show_shapes = True,
			   show_layer_names = False)
