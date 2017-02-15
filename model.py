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

    def __init__(self, img_height=256, img_width=256, train_flag=True):
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

    def create_model(self, train=True):
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
        y = Denormalize()(h)

        if not train:
            return Model(inputs, y)

        yc = Input((self.img_height, self.img_width, 3), name="contents_image")

        outputs = self.connect_vgg16(y, yc)
        self.model = Model([inputs, yc], outputs)
        return self.model

    def connect_vgg16(self, y, yc):
        vgg16 = VGG16(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(self.img_height, self.img_width, 3))

        # Frozen all layers of vgg16.
        for l in vgg16.layers:
            l.trainable = False

        vgg16.layers[ 2].name = "y1"
        vgg16.layers[ 5].name = "y2"
        vgg16.layers[ 9].name = "y3"
        vgg16.layers[13].name = "y4"

        # We need connect FastStyleNet and vgg16
        # to train FastStyleNet.
        h  = VGGNormalize(self.img_height, self.img_width, name="VGGNormalize")(y)
        h  = vgg16.layers[ 1](h)
        y1 = vgg16.layers[ 2](h)
        h  = vgg16.layers[ 3](y1)
        h  = vgg16.layers[ 4](h)
        y2 = vgg16.layers[ 5](h)
        h  = vgg16.layers[ 6](y2)
        h  = vgg16.layers[ 7](h)
        h  = vgg16.layers[ 8](h)
        y3 = vgg16.layers[ 9](h)
        h  = vgg16.layers[10](y3)
        h  = vgg16.layers[11](h)
        h  = vgg16.layers[12](h)
        y4 = vgg16.layers[13](h)

        h  = VGGNormalize(self.img_height, self.img_width, name="VGGNormalize2")(yc)
        h  = vgg16.layers[ 1](h)
        h  = vgg16.layers[ 2](h)
        h  = vgg16.layers[ 3](h)
        h  = vgg16.layers[ 4](h)
        h  = vgg16.layers[ 5](h)
        h  = vgg16.layers[ 6](h)
        h  = vgg16.layers[ 7](h)
        h  = vgg16.layers[ 8](h)
        yc3 = vgg16.layers[ 9](h)

        yc3 = merge(inputs=[yc3, y3],
                    output_shape=(64, 64, 256),
                    mode=lambda T: K.square(T[0]-T[1]))

        return  [y1, y2, y3, y4, yc3, y]

    def save_fastnet_weights(self, style_name, directory=None):
        '''
        Saves the weights of the FastNet model.
        It creates a temporary save file having the weights of FastNet + VGG,
        loads the weights into just the FastNet model and then deletes the
        FastNet + VGG weights.
        Args:
            style_name: style image name
            directory: base directory of saved weights
        '''
        import os

        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)

            full_weights_fn = directory + "temp.h5"
        else:
            full_weights_fn = "temp.h5"

        self.model.save_weights(filepath=full_weights_fn, overwrite=True)
        f = h5py.File(full_weights_fn)

        layer_names = [name for name in f.attrs['layer_names']]

        fsn_model = self.create_model(train=False)

        for i, layer in enumerate(fsn_model.layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        if directory is not None:
            weights_fn = directory + "{}.h5".format(style_name)
        else:
            weights_fn = "{}.h5".format(style_name)

        fsn_model.save_weights(weights_fn, overwrite=True)

        f.close()
        os.remove(full_weights_fn)  # The full weights aren't needed anymore since we only need 1 forward pass
                                    # through the fastnet now.
        print("Saved fastnet weights for style : %s.h5" % style_name)

if __name__ == "__main__":
    from keras.utils.visualize_util import plot
    fsn = FastStyleNet(train_flag=True)
    model = fsn.create_model()
    #plot(fsn.model, "fsn.png", show_shapes=True)
    plot(model, "train_model.png", show_shapes=True)
