# ensemble testing
# type "python hw6_test.py {test_x.csv} {dict.txt.big} {result.csv}" to execute

import sys
import time
import csv
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from gensim.models import KeyedVectors

import preprocess
import build_model

# ============================================

def output_result(result, output_path):
    mask = (result > 0.5)
    result[mask] = 1
    result[np.logical_not(mask)] = 0
    result = result.astype('int32')

    with open(output_path, 'w', newline = '') as fout:
        output = csv.writer(fout, delimiter = ',', lineterminator = '\n')
        output.writerow(['id', 'label'])
        ids = np.arange(len(result))[:, np.newaxis]
        rows = np.concatenate((ids, result), axis = 1)
        output.writerows(rows)

def ensemble_predict(x_test):
    results = list()

    model_weights_names = [x for x in os.listdir('./all_weights') if x.endswith('.h5')]
    for i in range(len(model_weights_names)):
        t = time.perf_counter()
        print('using model %d to predict...' % (i+1))
        K.clear_session()
        model = build_model.rnn_model(weights_path = 
                    os.path.join('./all_weights/', model_weights_names[i]))
        result = model.predict(x_test, batch_size = 2048)
        results.append(result)
        t = time.perf_counter() - t
        print('predict time: %.3f seconds.' % t)

    # averaging
    print('ensemble all results...')
    result = sum(results)/len(results)

    return result

def main(script, test_x, dict_txt, output_path):
    print('loading data and tokenizer...')
    x_test = preprocess.load_test(test_x)
    tokenizer = preprocess.create_tokenizer(None, './hw6_model/tokenizer')
    wv = KeyedVectors.load('./hw6_model/embedding.kv')

    print('tokenize and pad the data...')
    x_test = preprocess.tokenize(x_test, dict_txt)
    x_test = preprocess.replace_with_oov(x_test, wv)
    x_test = tokenizer.texts_to_sequences(x_test)
    max_len = preprocess.get_max_len()
    x_test = pad_sequences(x_test, maxlen = max_len)
    print('tokenize and padding finished!')

    print('predicting...')
    result = ensemble_predict(x_test)
    print('predict finished!')

    print('saving result')
    output_result(result, output_path)
    print('saved!')

# ============================================

if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t
    print('Executing time: %.3f seconds.' % t)
