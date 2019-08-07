# read "train.csv" and store the processed data as "./model/x_train.npy" and "./model/y_train.npy"

# type "python process_train_data.py {input_file}" to execute

# =========================================

import sys
import csv
import numpy as np

# =========================================

def load_train_data(train_file):
    '''
    Returns: A (18, 5760) list, representing each pollutant.
    '''
    ans = list()
    for i in range(18):
        ans.append(list())
    with open(train_file, 'r', newline = '', encoding = 'big5') as fin:
        rows = csv.reader(fin, delimiter = ',')
        rows.__next__() # skip the first row
        n_row = 0
        for row in rows:
            for element in row[3:]:
                if element == 'NR':
                    ans[n_row%18].append(float(0))
                else:
                    ans[n_row%18].append(float(element))
            n_row += 1
    return ans

def extract_feature(data):
    '''
    Returns: 
        x_train: A (5652, 162) numpy array.
        y_train: A (5652,) numpy array.
    '''
    x_train = list()
    y_train = list()
    for i in range(12): # 12 months
        for j in range(471): # 471 "consecutive 10 hours"
            x_train.append(list())
            y_train.append(data[9][480*i+j+9])
            for k in range(18):
                for m in range(9):
                    x_train[471*i+j].append(data[k][480*i+j+m])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

def main(script, train_file):
    data = load_train_data(train_file)
    x_train, y_train = extract_feature(data)
    np.save('./model/x_train.npy', x_train)
    np.save('./model/y_train.npy', y_train)

# =========================================

if __name__ == '__main__':
    main(*sys.argv)