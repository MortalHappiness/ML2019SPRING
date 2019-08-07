# use gensim to do the word embedding
# type "python embedding.py {train_x.csv} {test_x.csv} {dict.txt.big}" to execute

import time
import sys

from gensim.models import Word2Vec

import preprocess

# =============================================
def embedding(x_data):
    '''
    train the word embedding and save its KeyVectors as './hw6_model/embedding.kv'
    return the wordvector
    '''
    model = Word2Vec(x_data,
                     size = 200,
                     window = 5,
                     min_count = 6,
                     workers = 4,
                     iter = 5)
    model.wv.save('./hw6_model/embedding.kv')

    return model.wv

def main(script, train_x, test_x, dict_txt):
    print('loading data for embedding...')
    x_train = preprocess.load_train(train_x)
    x_test = preprocess.load_test(test_x)
    x_data = x_train + x_test
    x_data = preprocess.tokenize(x_data, dict_txt)
    print('data has been loaded!')

    print('embedding...')
    wv = embedding(x_data)
    print('embedding done!')

    print('replace the tokens that not in the word vectors with <oov>...')
    preprocess.replace_with_oov(x_data, wv)

    print('create tokenizer and save it...')
    tokenizer = preprocess.create_tokenizer(x_data)

# =============================================
if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t
    print('Executing time: %.3f seconds.' % t)
