# train the autoencoder
# type "python autoencoder.py" to execute

import os

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Deconv2D, MaxPooling2D, UpSampling2D, \
                         Activation, BatchNormalization, Flatten, Reshape, Concatenate, \
                         GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import utility

# ===================================

class AutoEncoder_Base:
    '''
    Base class for autoencoder, supporting training and testing.

    # Attributes:
        encoder(keras Model)
        decoder(keras Model)
        autoencoder(keras Model)
        batch_size(int)
        epochs(int)
    '''
    def __init__(self,
                 batch_size = 1024,
                 epochs = 20):
        '''
        # Arguments:
            structure(list of tuple): The structure from input tensor to latent space.
            batch_size(int)
            epochs(int)
        '''
        self.batch_size = batch_size
        self.epochs = epochs

    def build_encoder(self):
        raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError

    def build_autoencoder(self):
        raise NotImplementedError

    def train_on_generator(self, generator, images, save_folder):
        '''
        generator: keras ImageDataGenerator instance
        '''
        def pair_gen(data_gen):
            '''
            generate (x_train, y_train) pairs
            '''
            while True:
                batch = next(data_gen)
                yield batch, batch

        data_gen = generator.flow(images, batch_size = self.batch_size)

        model = self.autoencoder
        model.compile(loss = 'mse', optimizer = 'adam')

        model.fit_generator(
                    pair_gen(data_gen),
                    steps_per_epoch = len(images)//self.batch_size,
                    epochs = self.epochs
                    )

        # only save weights for encoder and autoencoder
        model.save_weights(os.path.join(save_folder, 'autoencoder.h5'))
        self.encoder.save_weights(os.path.join(save_folder, 'encoder.h5'))

    def load_weights(self, weight_folder):
        self.autoencoder.load_weights(os.path.join(weight_folder, 'autoencoder.h5'))
        self.encoder.load_weights(os.path.join(weight_folder, 'encoder.h5'))

    def img_to_latent(self, images):
        '''
        Use encoder to encode the images.
        '''
        latent = self.encoder.predict(images, batch_size = self.batch_size)
        return latent.reshape(len(latent), -1)

    def img_to_img(self, images):
        '''
        Feed the images into autoencoder to see the output images.
        '''
        return self.autoencoder.predict(images, batch_size = self.batch_size)

class ConvAutoEncoder(AutoEncoder_Base):
    '''
    Build the convlutional autoencoder.

    # Attributes:
        encoder(keras Model)
        decoder(keras Model)
        autoencoder(keras Model)
        batch_size(int)
        epochs(int)
    '''
    def __init__(self,
                 structure = [(32, 32, 3),
                              ('conv', (3, 3), 64),
                              ('pool',),
                              ('conv', (1, 1), 8)],
                 batch_size = 2048,
                 epochs = 10):
        '''
        # Arguments:
            structure(list of tuple): The structure from input tensor to latent space.
            batch_size(int)
            epochs(int)
        '''
        super(ConvAutoEncoder, self).__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self._input_shape = structure[0]
        self._encoder_structure = structure[1:]

        # translate the encoder structure into decoder structure
        decoder_structure = list()
        filters = 3
        for x in structure[1:]:
            if x[0] == 'conv':
                decoder_structure.append(('deconv', x[1], filters))
                filters = x[2]
            elif x[0] == 'pool':
                decoder_structure.append(('uppool',))
            else:
                raise Exception

        self._decoder_structure = decoder_structure[::-1]

        self.build_autoencoder()

    def build_encoder(self):
        self._input_tensor = Input(shape = self._input_shape)
        x = self._input_tensor
        for structure in self._encoder_structure:
            if structure[0] == 'conv':
                x = Conv2D(filters = structure[2],
                           kernel_size = structure[1],
                           padding = 'same',
                           use_bias = False)(x)
            elif structure[0] == 'pool':
                x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)
            else:
                raise Exception
        latent = x

        self.encoder = Model(self._input_tensor, latent, name = 'encoder')

        self._latent_shape = K.int_shape(latent)[1:]

    def build_decoder(self):
        decoder_input = Input(shape = self._latent_shape)
        x = decoder_input
        for structure in self._decoder_structure:
            if structure[0] == 'deconv':
                x = Deconv2D(filters = structure[2],
                                 kernel_size = structure[1],
                                 padding = 'same',
                                 use_bias = False)(x)
            elif structure[0] == 'uppool':
                x = UpSampling2D(size = (2, 2))(x)
            else:
                raise Exception
        x = Activation('sigmoid')(x)
        output = x

        self.decoder = Model(decoder_input, output, name = 'decoder')

    def build_autoencoder(self):
        self.build_encoder()
        self.build_decoder()
        self.autoencoder = Model(self._input_tensor,
                                 self.decoder(self.encoder(self._input_tensor)),
                                 name = 'autoencoder')

# ===================================

AutoEncoder = ConvAutoEncoder

# ===================================

if __name__ == '__main__':
    if os.path.exists('./data/images.npy'):
        images = np.load('./data/images.npy')
    else:
        images = utility.load_image('./data/images/')

    images = images/255

    generator = ImageDataGenerator(
                # featurewise_center = True,
                # featurewise_std_normalization = True,
                rotation_range = 30,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = [0.75, 1.25],
                horizontal_flip = True)
    generator.fit(images)

    model = AutoEncoder(batch_size = 1024, epochs = 30)
    model.encoder.summary()
    model.decoder.summary()
    model.train_on_generator(generator, images, save_folder = './model/')