# type "python hw4.py {train.csv} {result/}" to execute

import sys
import csv
import os
import time

import numpy as np
from keras.utils import to_categorical
from keras import backend as K

from PIL import Image
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.color import gray2rgb, rgb2gray
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from my_model import build_model

# =====================================

np.random.seed(1000)

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

    x_train = [None] * 7
    y_train = list(range(7))

    idx = [28200, 28234, 27870, 28699, 28194, 28705, 28175] # row in train.csv
    idx = np.array(idx)
    idx -= 2 # skip the header and let first row be 0

    with open(train_data_path, 'r', newline = '') as fin:
        rows = csv.reader(fin)
        next(rows) # skip the header
        num = 0
        for row in rows:
            if num not in idx:
                num += 1
                continue
            num += 1
            label, feature = row
            label = int(label)
            feature = [int(x) for x in feature.split()]
            feature = np.array(feature).reshape(48, 48)
            x_train[label] = feature
    x_train = np.array(x_train).astype('float32')/255
    y_train = np.array(y_train)

    x_train = x_train[:, :, :, np.newaxis]
    y_train = to_categorical(y_train)
    
#     # original image
#     for i in range(7):
#         im = deprocess_image(x_train[i].reshape(48, 48))
#         plt.figure(figsize = (5, 5))
#         plt.imshow(im, cmap = plt.cm.gray)
#         plt.savefig(f'./original/{i}.jpg')

    print('training data loading finished')

    return x_train, y_train

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def saliency_map(model, images, output_folder):
    for i in range(7):
        image = K.variable(images[i][np.newaxis, :, :, :])
        pred = model(image)
        grad = K.gradients(pred[0, i], image)[0]
        arr = np.abs(K.eval(grad).reshape(48, 48))
        arr = deprocess_image(arr)

        plt.figure(figsize = (5, 5))
        plt.imshow(arr, cmap = plt.cm.jet)
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, f'fig1_{i}.jpg'))
        
#         # heat map
#         img = images[i].reshape(48, 48)
#         img[arr < 128] = np.mean(img)
#         img = np.clip(img*255, 0, 255)
#         plt.figure(figsize = (5, 5))
#         plt.imshow(img, cmap = plt.cm.gray)
#         plt.colorbar()
#         plt.savefig(os.path.join(output_folder, f'{i}.jpg'))
        
        
def filter_visualize(model, images, output_folder):
    input_img = model.inputs[0]
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    layer_name = 'conv2d_2'
    filter_indices = list(range(32))
    
    plt.figure(figsize = (16, 8))
    for i in range(len(filter_indices)):
        filter_index = filter_indices[i]
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_img], [loss, grads])

        input_img_data = np.zeros((1, 48, 48, 1))

        for j in range(40):
            loss, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 30

        img = input_img_data[0]
        img = deprocess_image(img).reshape(48, 48)

        plt.subplot2grid((4, 8), (i//8, i % 8))
        plt.imshow(img)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.suptitle('The first 32 filters of %s' % layer_name, fontsize = 20)
    plt.savefig(os.path.join(output_folder, 'fig2_1.jpg'))
    
    # input an image, observe its output
    layer_output = layer_dict[layer_name].output[:, :, :, 0:32]
    func = K.function([input_img], [layer_output])
    outputs = func([images[0].reshape(1, 48, 48, 1)])
    
    plt.figure(figsize = (16, 8))
    for i in range(32):
        img = deprocess_image(outputs[0][:, :, :, i]).reshape(24, 24)
        plt.subplot2grid((4, 8), (i//8, i % 8))
        plt.imshow(img)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.suptitle('The output of the first 32 filters of %s' % layer_name, fontsize = 20)
    plt.savefig(os.path.join(output_folder, 'fig2_2.jpg'))
        
def show_lime(model, x_data, output_folder):
    x_data = np.clip(x_data * 255, 0, 255).reshape(7, 48, 48).astype('uint8')
    x_data_rgb = gray2rgb(x_data)
    x_label = np.arange(7, dtype = 'float32')
    
    def predict(image):
        image = rgb2gray(image).reshape(-1, 48, 48, 1)
        return model.predict(image)

    def segmentation(image):
        return slic(image)

    # Initiate explainer instance
    explainer = lime_image.LimeImageExplainer()

    for idx in range(7):
        # Get the explaination of an image
        np.random.seed(1000)
        explaination = explainer.explain_instance(
                                    image = x_data_rgb[idx], 
                                    classifier_fn = predict,
                                    segmentation_fn = segmentation
                                )

        # Get processed image
        image, mask = explaination.get_image_and_mask(
                                        label = x_label[idx],
                                        positive_only = False,
                                        hide_rest = False,
                                        num_features = 5,
                                        min_weight = 0.0
                                    )
        # save the image
        plt.figure(figsize = (5, 5))
        plt.imshow(mark_boundaries(image, mask))
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.savefig(os.path.join(output_folder, f'fig3_{idx}.jpg'))
    
def main(script, train_data_path, output_folder):
    x_data, y_data = load_train_data(train_data_path) # len: 28709

    model = build_model()
    model.load_weights('./model.h5')
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    saliency_map(model, x_data, output_folder)
    filter_visualize(model, x_data, output_folder)
    show_lime(model, x_data, output_folder)

# ====================================

if __name__ == '__main__':
    t = time.perf_counter()
    main(*sys.argv)
    t = time.perf_counter() - t

    print('executing time: %.3f seconds' % t)
