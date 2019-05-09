# utility functions for data preprocessing

import csv
import time
import re
import string
import shelve
from multiprocessing import Pool

from keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd
import jieba

# ========================================

def load_train(train_path):
	'''
	load the training text

	# Arguments:
		train_path(str): path to train_x.csv

	# Returns:
		sentences(list of str): a list containing sentences
	'''
	sentences = list()
	with open(train_path, 'r', newline = '') as fin:
		rows = csv.reader(fin, delimiter = ',', lineterminator = '\n')
		next(rows) # skip the header
		for row in rows:
			sentences.append(row[1])
	return sentences[0:119018] # the sentences out of this range are repeated

def load_test(test_path):
	'''
	load the testing text

	# Arguments:
		test_path(str): path to test_x.csv

	# Returns:
		sentences(list of str): a list containing sentences
	'''
	sentences = list()
	with open(test_path, 'r', newline = '') as fin:
		rows = csv.reader(fin, delimiter = ',', lineterminator = '\n')
		next(rows) # skip the header
		for row in rows:
			sentences.append(row[1])
	return sentences

def load_label(label_path):
	'''
	load the training label

	# Arguments:
		label_path(str): path to train_y.csv

	# Returns:
		y_label(ndarray): an array contains all labels
	'''
	y_label = pd.read_csv(label_path)['label'].to_numpy().astype('float32')
	y_label = y_label[0:119018] # the sentences out of this range are repeated

	return y_label
	
def preprocess(sentence):
	# make all english characters in the sentence be all lowercase
	sentence = sentence.lower()

	# remove all symbols, emojis, punctuations
	# sentence = re.sub(r'(?u)([^\w\s])+', ' ', sentence)

	# remove "bxxx", where xxx are consecutive digits
	sentence = re.sub(r'b\d+', ' ', sentence)

	# replace continuous whitespaces with a single space
	sentence = re.sub(r'\s+', ' ', sentence)

	# replace continuous repeated words with a single word
	# pattern = re.compile(r'(( \w+)\2+)')
	# cont = pattern.findall(sentence)
	# if cont: 
	# 	sentence = sentence.replace(*cont[0])

	# replace consecutive characters with a single character
	# for repeat_char, char in re.findall(r'((\w)\2+)', sentence):
	# 	sentence = sentence.replace(repeat_char, char)

	return sentence

def cut_sentence(sentence):
	sentence = preprocess(sentence)
	seg_list = list(jieba.cut(sentence))
	seg_list = [x for x in seg_list if not x.isspace()]
	return seg_list

def tokenize(x_data, dict_path):
	'''
	Tokenize every sentences in x_data

	# Arguments:
		x_data(list of str): a list containing all the sentences
		dict_path(str): path to dict.txt.big

	# Returns:
		x_tokens(list of list of str): list of list of tokens
	'''
	jieba.load_userdict(dict_path)

	with Pool(processes = 4) as pool:
		x_tokens = pool.map(cut_sentence, x_data)

	return x_tokens

def create_tokenizer(x_tokens, tokenizer_path = None):
	'''
	Train a keras Tokenizer instance if tokenizer_path is not None,
	else return the pretrained tokenizer.

	# Arguments:
		x_tokens(list of list of str): list of list of tokens
		tokenizer_path(str): path to tokenizer database

	# Returns:
		tokenizer(Tokenizer): keras Tokenizer instance
	'''
	if tokenizer_path != None:
		with shelve.open(tokenizer_path) as db:
			return db['tokenizer']

	tokenizer = Tokenizer(num_words = None, filters = '')
	tokenizer.fit_on_texts(x_tokens)
	word_index = tokenizer.word_index
	print('Total tokens:', len(word_index))

	print('saving the tokenizer')
	with shelve.open('./hw6_model/tokenizer') as db:
		db['tokenizer'] = tokenizer
	print('tokenizer has been saved!')

	return tokenizer

def replace_with_oov(x_data, wv):
	# replace the tokens that not in the word vectors with <oov>
	oov_cnt = 0
	for i in range(len(x_data)):
		for j in range(len(x_data[i])):
			if x_data[i][j] not in wv.vocab:
				# print(x_data[i][j], 'not in word vectors.')
				x_data[i][j] = '<oov>'
				oov_cnt += 1
	print('oov_cnt:', oov_cnt)

	return x_data

def get_max_len():
	'''
	Return the max length of the input sequence.
	'''
	with open('./hw6_model/max_len.txt', 'r') as fin:
		max_len = int(fin.read().strip())
	return max_len

# ========================================

if __name__ == '__main__':
	t = time.perf_counter()

	x_train = load_train('./data/train_x.csv')
	x_test = load_test('./data/test_x.csv')
	y_label = load_label('./data/train_y.csv')
	x_data = x_train + x_test
	x_data = tokenize(x_data, './data/dict.txt.big')

	tokenizer = create_tokenizer(x_data)

	t = time.perf_counter() - t
	print('Executing time: %.3f seconds.' % t)