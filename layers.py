from keras.engine.topology import Layer
from keras import backend as K

'''Referred to "https://github.com/titu1994/Fast-Neural-Style/blob/master/layers.py"
'''
class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)
    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
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

class SubMean(Layer):
    def __init__(self, **kwargs):
        super(SubMean, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x - 120

    def get_output_shape_for(self, input_shape):
        return input_shape

class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # No exact substitute for set_subtensor in tensorflow
        # So we subtract an approximate value
        r,g,b = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]
        r -= 103.939
        g -= 116.779
        b -= 123.68
        # 'RGB'->'BGR'
        r,g,b = K.expand_dims(r), K.expand_dims(g), K.expand_dims(b)
        cx = K.concatenate([b,g,r], axis=-1)
        return cx

    def get_output_shape_for(self, input_shape):
        return input_shape
