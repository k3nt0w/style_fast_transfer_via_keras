from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.applications.vgg16 import VGG16

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256,256,3))

class ResidualBlock(Layer):
    def __init__(self, nb_filter, nb_row, nb_col, subsample=(1,1), **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.subsample = subsample
        super(ResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        # This layer is just stacking layers,
        # so we do not need about "weight" implementation.
        pass

    def call(self, x, mask=None):
        h = Convolution2D(self.nb_filter, self.nb_row, self.nb_col, subsample=self.subsample, border_mode='same')(x)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)
        h = Convolution2D(self.nb_filter, self.nb_row, self.nb_col, subsample=(1, 1), border_mode='same')(h)
        h = BatchNormalization()(h)
        return h + x

    def get_output_shape_for(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]
        return (input_shape[0], rows, cols, self.nb_filter)

def FastStyleNet():
    # use "tf" dim-ordering
    inputs = Input((256,256,3))

    h = Convolution2D(32, 9, 9, border_mode='same')(inputs)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = Convolution2D(64, 4, 4, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = Convolution2D(128, 4, 4, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    h = ResidualBlock(128, 3, 3)(h)
    h = ResidualBlock(128, 3, 3)(h)
    h = ResidualBlock(128, 3, 3)(h)
    h = ResidualBlock(128, 3, 3)(h)
    h = ResidualBlock(128, 3, 3)(h)

    #h = UpSampling2D()(h)
    h = Convolution2D(64, 3, 3, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    #h = UpSampling2D()(h)
    h = Convolution2D(32, 3, 3, border_mode='same')(h)
    h = BatchNormalization()(h)
    h = ELU()(h)

    out = Convolution2D( 3, 9, 9, border_mode='same')(h)

    return Model(inputs, out)

def connect_vgg16():
    # We need connect FastStyleNet and vgg16
    # to train FastStyleNet.
    fsn = FastStyleNet()
    fsn.name = "FastStyleNet"

    # Frozen all layers of vgg.
    for l in vgg16.layers:
        l.trainable = False

    vgg16.layers[2].name = "y1"
    vgg16.layers[5].name = "y2"
    vgg16.layers[9].name = "y3"
    vgg16.layers[13].name = "y4"

    inputs = Input((256, 256, 3), name="inputs")
    tv = fsn(inputs)
    h  = vgg16.layers[1](tv)
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

    model = Model(inputs, output= [y1, y2, y3, y4, y3, tv])
    return model, fsn

if __name__ == "__main__":
    from keras.utils.visualize_util import plot
    model, fsn = connect_vgg16()
    plot(fsn, "fsn.png", show_shapes=True)
    plot(model, "train_model.png", show_shapes=True)
    print(model.summary())