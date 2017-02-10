from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Activation
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.applications.vgg16 import VGG16

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256,256,3))

def residual_block(x, nb_filter, ksize):
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


def FastStyleNet():
    # use "tf" dim-ordering
    inputs = Input((None, None, 3))

    h = Convolution2D(32, 9, 9, border_mode='same')(inputs)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = Convolution2D(64, 4, 4, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = Convolution2D(128, 4, 4, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = residual_block(h, 128, 3)
    h = residual_block(h, 128, 3)
    h = residual_block(h, 128, 3)
    h = residual_block(h, 128, 3)
    h = residual_block(h, 128, 3)

    h = Convolution2D(64, 3, 3, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = Convolution2D(32, 3, 3, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    out = Convolution2D( 3, 9, 9, border_mode='same')(h)

    return Model(inputs, out)

def connect_vgg16(size):
    # We need connect FastStyleNet and vgg16
    # to train FastStyleNet.
    fsn = FastStyleNet()
    fsn.name = "FastStyleNet"

    # Frozen all layers of vgg16.
    for l in vgg16.layers:
        l.trainable = False

    vgg16.layers[2].name = "y1"
    vgg16.layers[5].name = "y2"
    vgg16.layers[9].name = "y3"
    vgg16.layers[13].name = "y4"

    ip = Input((size, size, 3), name="input")
    for_tv = fsn(ip)

    # I think that it can be done more easily here...
    # Please give me some idea.
    h  = vgg16.layers[1](for_tv)
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

    h  = vgg16.layers[1](ip)
    h  = vgg16.layers[2](h)
    h  = vgg16.layers[3](h)
    h  = vgg16.layers[4](h)
    h  = vgg16.layers[5](h)
    h  = vgg16.layers[6](h)
    h  = vgg16.layers[7](h)
    h  = vgg16.layers[8](h)
    cy3 = vgg16.layers[9](h)

    # Calculating the square error in this layer has no problem.
    cy3 = merge(inputs=[cy3, y3],
                output_shape=(None, 64, 64, 1),
                mode=lambda T: K.square(T[0][0,:,:,:]-T[1][0,:,:,:]))

    model = Model(input=ip, output=[y1, y2, y3, y4, cy3, for_tv])
    return model, fsn

if __name__ == "__main__":
    from keras.utils.visualize_util import plot
    model, _ = connect_vgg16()
    model.summary()
    plot(model, "train_model.png")
