# train
# type "python hw3_train.py {train.csv}" to execute

import sys
import csv
import os
import shutil

import numpy as np

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from my_model import build_model

# ====================================
vali = 1 # bool, whether to use validation data
aug = 1 # bool, whether to use data augmentation

batch_size = 128
epochs = 400

callbacks = [ModelCheckpoint('./checkpoint/weights_{epoch:03d}_{loss:.2f}_{val_loss:.2f}_{acc:.2f}_{val_acc:.2f}.h5',
							 save_weights_only = True,
							 period = 1)
			]
if os.path.exists('./checkpoint'):
	shutil.rmtree('./checkpoint')
os.mkdir('./checkpoint')

# ====================================
def load_train_data(train_data_path):
	'''
	Load the training data.

	# Arguments:
		train_data_path(str): The path to train.csv

	# Returns:
		x_train(array): shape = (28709, 48, 48, 1)
		y_train(array): shape = (28709, 7)
	'''
	print('loading the training data ...')

	x_train = list()
	y_train = list()
	with open(train_data_path, 'r', newline = '') as fin:
		rows = csv.reader(fin)
		next(rows) # skip the header
		for row in rows:
			label, feature = row
			feature = [int(x) for x in feature.split()]
			feature = np.array(feature).reshape(48, 48)
			x_train.append(feature)
			y_train.append(int(label))
	x_train = np.array(x_train).astype('float32')/255
	y_train = np.array(y_train)

	x_train = x_train[:, :, :, np.newaxis]
	y_train = to_categorical(y_train)

	print('training data loading finished')

	return x_train, y_train

def train_model(model, x_train, y_train, validation_data):
	model.compile(loss = 'categorical_crossentropy',
				  optimizer = Adam(),
				  metrics = ['accuracy'])

	if aug:
		generator = ImageDataGenerator(
						# featurewise_center = True,
						# featurewise_std_normalization = True,
						rotation_range = 30,
						width_shift_range = 0.2,
						height_shift_range = 0.2,
						shear_range = 0.2,
						zoom_range = [0.75, 1.25],
						horizontal_flip = True)
		generator.fit(x_train)

		model.fit_generator(
					generator.flow(x_train, y_train, batch_size = batch_size),
					steps_per_epoch = 4*len(x_train)//batch_size,
					epochs = epochs,
					validation_data = validation_data,
					callbacks = callbacks
					)
	else: # no aug
		model.fit(x_train, y_train,
					  batch_size = batch_size,
					  epochs = epochs,
					  validation_data = validation_data,
					  callbacks = callbacks)

	model.save_weights('./hw3_model/model.h5')

	if vali:
		print('train acc: %.3f' % model.evaluate(x_train, y_train)[1])
		print('test acc: %.3f' % model.evaluate(*validation_data)[1])

def main(script, train_data_path):
	x_data, y_data = load_train_data(train_data_path) # len: 28709

	if vali:
		train_idx = np.random.permutation(np.arange(len(x_data))) < 22000
		test_idx = np.logical_not(train_idx)
		x_train = x_data[train_idx]
		y_train = y_data[train_idx]
		x_test = x_data[test_idx]
		y_test = y_data[test_idx]
		validation_data = (x_test, y_test)
	else:
		x_train = x_data
		y_train = y_data
		validation_data = None

	model = build_model()
	train_model(model, x_train, y_train, validation_data)

# ====================================

if __name__ == '__main__':
	main(*sys.argv)
