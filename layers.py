from keras.engine.topology import Layer
from keras import backend as K

import numpy as np


class Denormalize(Layer):
    '''Referred to "https://github.com/titu1994/Fast-Neural-Style/blob/master/layers.py"
    '''
    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def get_output_shape_for(self, input_shape):
        return input_shape

class VGGNormalize(Layer):
    """Prepare for extract features from vgg16
    """
    def __init__(self, height, width, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.MEAN_VALUE_TENSOR = self.make_mean_tensor()

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # 'RGB'->'BGR'
        x = K.reverse(x, axes=-1)
        # Zero-center by mean pixel
        x = x - self.MEAN_VALUE_TENSOR
        return x

    def make_mean_tensor(self):
        x = np.empty((1,self.height,self.width,3))
        x[:, :, :, 0] = 103.939
        x[:, :, :, 1] = 116.779
        x[:, :, :, 2] = 123.68
        return K.variable(x)

    def get_output_shape_for(self, input_shape):
        return input_shape
