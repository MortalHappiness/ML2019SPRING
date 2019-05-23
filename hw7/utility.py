# utility functions

import os
import time

import numpy as np
from PIL import Image

# ===============================
def load_image(img_folder):
	'''
	Load the images as numpy array with dtype uint8(without normalization).
	'''
	t = time.perf_counter()
	print('loading images...')

	images = list()
	for i in range(1, 40001):
		if i % 1000 == 0:
			print('\t%d images have been loaded.' % i)
		with Image.open(os.path.join(img_folder, str(i).rjust(6, '0') + '.jpg')) as img:
			images.append(np.asarray(img))
	images = np.array(images)

	t = time.perf_counter() - t
	print('images loading time: %.3f seconds' % t)

	return images

# ===============================
if __name__ == '__main__':
	images = load_image('./data/images/')
	print(images.shape)
	print(images.dtype)
	np.save('./data/images.npy', images)
