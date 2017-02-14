from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Activation
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Deconvolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input

from layers import *

import h5py
import numpy as np

class FastStyleNet:

    def __init__(self, img_height=256, img_width=256):
        self.img_height = img_height
        self.img_width = img_width

    def residual_block(sefl, x, nb_filter, ksize):
        h = Convolution2D(nb_filter, ksize, ksize, subsample=(1, 1), border_mode='same')(x)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)
        h = Convolution2D(nb_filter, ksize, ksize, subsample=(1, 1), border_mode='same')(h)
        h = BatchNormalization()(h)
        if K.int_shape(x) != K.int_shape(h):
            n, c, hh, ww = K.int_shape(x)
            pad_c = K.int_shape(h)[1] - c
            p = K.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = K.variable(p)
            x = K.concatenate([p, x], axis=1) #channel
            if K.int_shape(x)[2:] != K.int_shape(h)[2:]:
                x = AveragePooling2D(pool_size=(2, 2), strides=1)(x)
        m = merge([h, x], mode='sum')
        return m

    def create_model(self):
        # use "tf" dim-ordering
        inputs = Input((self.img_height, self.img_width, 3))

        h = Convolution2D(32, 9, 9, border_mode="same")(inputs)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = ZeroPadding2D((1, 1))(h)
        h = Convolution2D(64, 4, 4, border_mode='valid', subsample=(2, 2))(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = ZeroPadding2D((1, 1))(h)
        h = Convolution2D(128, 4, 4, border_mode='valid', subsample=(2, 2))(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = self.residual_block(h, 128, 3)
        h = self.residual_block(h, 128, 3)
        h = self.residual_block(h, 128, 3)
        h = self.residual_block(h, 128, 3)
        h = self.residual_block(h, 128, 3)

        h = Deconvolution2D(64, 4, 4, activation="linear", border_mode="same", subsample=(2, 2),
                                 output_shape=(1, self.img_height // 2, self.img_width // 2, 64),
                                 name="deconv3")(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Deconvolution2D(32, 4, 4, activation="linear", border_mode="same", subsample=(2, 2),
                                 output_shape=(1, self.img_height, self.img_width, 32),
                                 name="deconv2")(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Deconvolution2D(3, 9, 9, activation="tanh", border_mode="same", subsample=(1, 1),
                                 output_shape=(1, self.img_height, self.img_width, 3),
                                 name="deconv1")(h)
        out = Denormalize()(h)

        return Model(inputs, out)

    def connect_vgg16(self):
        vgg16 = VGG16(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(self.img_height, self.img_width, 3))

        vgg16_2 = VGG16(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(self.img_height, self.img_width, 3))

        namelist = list()
        for l in vgg16.layers:
            namelist.append(l.name)

        for i, l in enumerate(vgg16_2.layers):
            l.name = "{}_2".format(namelist[i])

        # Frozen all layers of vgg16.
        for l, l2 in zip(vgg16.layers, vgg16_2.layers):
            l.trainable = False
            l2.trainable = False

        vgg16.layers[ 2].name = "y1"
        vgg16.layers[ 5].name = "y2"
        vgg16.layers[ 9].name = "y3"
        vgg16.layers[13].name = "y4"

        # We need connect FastStyleNet and vgg16
        # to train FastStyleNet.
        fsn = self.create_model()
        fsn.name = "FastStyleNet"

        ip1 = Input((self.img_height, self.img_width, 3), name="input1")
        y0 = fsn(ip1)
        cy0 = VGGNormalize(self.img_height, self.img_width)(y0)

        """I think that it can be done more easily here...
        Please give me some idea."""
        h  = vgg16.layers[1](cy0)
        y1 = vgg16.layers[2](h)
        h  = vgg16.layers[3](y1)
        h  = vgg16.layers[4](h)
        y2 = vgg16.layers[5](h)
        h  = vgg16.layers[6](y2)
        h  = vgg16.layers[7](h)
        h  = vgg16.layers[8](h)
        y3 = vgg16.layers[9](h)
        h  = vgg16.layers[10](y3)
        h  = vgg16.layers[11](h)
        h  = vgg16.layers[12](h)
        y4 = vgg16.layers[13](h)

        # to get contents featuer from conv3_3
        ip2 = Input((self.img_height, self.img_width, 3), name="input2")
        h = VGGNormalize(self.img_height, self.img_width)(ip2)
        h = vgg16_2.layers[1](h)
        h = vgg16_2.layers[2](h)
        h = vgg16_2.layers[3](h)
        h = vgg16_2.layers[4](h)
        h = vgg16_2.layers[5](h)
        h = vgg16_2.layers[6](h)
        h = vgg16_2.layers[7](h)
        h = vgg16_2.layers[8](h)
        cy3 = vgg16_2.layers[9](h)

        # Calculating the square error in this layer has no problem.
        cy3 = merge(inputs=[cy3, y3],
                    output_shape=(64, 64, 256),
                    mode=lambda T: K.square(T[0]-T[1]))

        train_model = Model(input=[ip1,ip2], output=[y1, y2, y3, y4, cy3, y0])
        return train_model, fsn

if __name__ == "__main__":
    from keras.utils.visualize_util import plot
    fsn = FastStyleNet()
    model, fsn_model = fsn.connect_vgg16()
    model.summary()
    plot(fsn_model, "fsn.png", show_shapes=True)
    plot(model, "train_model.png", show_shapes=True)
