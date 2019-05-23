# cluster
# type "python cluster.py {images} {test_case.csv} {output_path}" to execute

import time
import sys
import csv

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utility import load_image
from autoencoder import AutoEncoder

# =====================================

def load_test_case(test_path):
	test_df = pd.read_csv(test_path)
	return test_df[['image1_name', 'image2_name']].to_numpy() - 1

def output_result(predict, output_path):
	with open(output_path, 'w', newline = '') as fin:
		output_file = csv.writer(fin)
		output_file.writerow(['id', 'label'])
		ids = np.arange(len(predict))[:, np.newaxis]
		predict = predict[:, np.newaxis]
		rows = np.concatenate((ids, predict), axis = 1)
		output_file.writerows(rows)

def cluster(model, images):
	latent = model.img_to_latent(images)
	print('The latent shape of the encoder output:', latent.shape)

	t = time.perf_counter()
	pca = PCA(n_components = 400,
			  whiten = True,
			  svd_solver = 'full',
			  random_state = 0)
	latent = pca.fit_transform(latent)
	print('The latent shape after PCA:', latent.shape)
	t = time.perf_counter() - t
	print('PCA time: %.3f seconds' % t)

	t = time.perf_counter()
	cluster = KMeans(n_clusters = 2, random_state = 0).fit(latent)
	labels = cluster.labels_
	t = time.perf_counter() - t
	print('Clustering time: %.3f seconds' % t)

	return labels

def main(script, img_folder, test_case, output_path):
	model = AutoEncoder()
	model.load_weights('./model/')
	images = load_image(img_folder) / 255
	labels = cluster(model, images)
	
	test_case = load_test_case(test_case)
	test_label = labels[test_case]
	predict = list()
	for label1, label2 in zip(test_label[:, 0], test_label[:, 1]):
		if label1 == label2:
			predict.append(1)
		else:
			predict.append(0)
	predict = np.array(predict, dtype = int)
	output_result(predict, output_path)

# ===============================

if __name__ == '__main__':
	t = time.perf_counter()
	main(*sys.argv)
	t = time.perf_counter() - t
	print('Execute time: %.3f seconds.' % t)