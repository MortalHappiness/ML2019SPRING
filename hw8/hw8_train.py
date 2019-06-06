# training
# type "python hw8_train.py {train.csv}" to execute

import sys
import csv

import numpy as np

import load_data
from my_model import CompressedModel

# ======================================

def main(script, train_data_path):
	x_data, y_data = load_data.load_train_data(train_data_path) # len: 28709

	train_idx = np.arange(len(x_data)) < 27000
	test_idx = np.logical_not(train_idx)
	x_train = x_data[train_idx]
	y_train = y_data[train_idx]
	x_test = x_data[test_idx]
	y_test = y_data[test_idx]

	model = CompressedModel(epochs = 2000)
	print('distill...')
	model.distill(x_train, y_train, x_test, y_test)

# ====================================

if __name__ == '__main__':
	main(*sys.argv)
