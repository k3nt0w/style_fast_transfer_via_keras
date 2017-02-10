from model import *
from keras import backend as K

import numpy as np
from PIL import Image
import os

def load_image(path, size):
    image = Image.open(path).convert('RGB')
    w,h = image.size
    if w < h:
        if w < size:
            image = image.resize((size, size*h//w))
            w, h = image.size
    else:
        if h < size:
            image = image.resize((size*w//h, size))
            w, h = image.size
    image = image.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))
    image = np.asarray(image, dtype=np.float32)
    return image[np.newaxis, :]

def get_style_features(style_img):
    inputs  = K.variable(style_img)
    h  = vgg16.layers[ 0](inputs)
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
    return [y1, y2, y3, y4]

def get_contents_features(contents_img):
    inputs  = K.variable(contents_img)
    h  = vgg16.layers[ 0](inputs)
    h  = vgg16.layers[ 1](h)
    h  = vgg16.layers[ 2](h)
    h  = vgg16.layers[ 3](h)
    h  = vgg16.layers[ 4](h)
    h  = vgg16.layers[ 5](h)
    h  = vgg16.layers[ 6](h)
    h  = vgg16.layers[ 7](h)
    h  = vgg16.layers[ 8](h)
    y3 = vgg16.layers[ 9](h)
    return y3
