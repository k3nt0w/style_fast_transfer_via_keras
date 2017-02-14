from model import *
from keras import backend as K

import numpy as np
from PIL import Image
import os

def load_image(path, size):
    img = Image.open(path).convert('RGB')
    w,h = img.size
    if w < h:
        if w < size:
            img = img.resize((size, size*h//w))
            w, h = img.size
    else:
        if h < size:
            img = img.resize((size*w//h, size))
            w, h = img.size
    img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0) #(1, h, w, ch)
    return x

def get_style_features(style_img, height, width):
    vgg = vgg16 = VGG16(include_top=False,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=(height, width, 3))

    inputs = K.variable(style_img)
    h  = vgg16.layers[ 1](inputs)
    y1 = vgg16.layers[ 2](h)
    h  = vgg16.layers[ 3](h)
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

    return [y1, y2, y3, y4]

def preprocess_input(x, dim_ordering='default'):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        dim_ordering: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x
