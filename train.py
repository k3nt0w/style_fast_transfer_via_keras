from model import *
from preprocess import *
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import argparse
import os
import sys
from multiprocessing import Pool, Process


parser = argparse.ArgumentParser(description='Real-time style transfer via Keras')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--weight', '-w', default="", type=str)
parser.add_argument('--lambda_tv', default=1e-6, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', default=1.0, type=float)
parser.add_argument('--lambda_style', default=5.0, type=float)
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--image_size', default=256, type=int)

args = parser.parse_args()

image_size = args.image_size
nb_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style

def gram_matrix(x):
    """I reffered to
    "https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py"
    """
    x = x[0,:,:,:] # (row, col, ch)

    nrow, ncol, nch = K.int_shape(x)

    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features)) / (nrow * ncol * nch)
    return gram

def style_reconstruction_loss(gram_s):
    """we should calculate gram matrix of style image just once.
    Therefore, I implemented this function like this.
    """
    def loss_function(y_true, y_pred):
        gram_s_hat = gram_matrix(y_pred)
        return lambda_s * K.mean(K.square(gram_s_hat - gram_s))
    return loss_function

def feature_reconstruction_loss(y_true, y_pred):
    """This function will receive a tensor that
    already calculated the square error.
    So, just calculate the average
    """
    return lambda_f * K.mean(y_pred)

def total_variation_loss(y_true, x):
    """I reffered to
    "https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py"
    """
    assert K.ndim(x) == 4
    img_nrows = image_size
    img_ncols = image_size
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return lambda_tv*K.mean(K.pow(a + b, 1.25))

# ---------------- train --------------------

""" "imagepaths" is a list containing absolute paths of Dataset.
"""
fs = os.listdir(args.dataset)
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)

nb_data = len(imagepaths)

model, fsn = connect_vgg16()
if len(args.weight) > 0:
    fsn.load_weights(args.weight)
style_img = load_image(args.style_image, args.image_size)
contents_img = load_image(imagepaths[0], args.image_size)

style_features = get_style_features(style_img)
y1, y2, y3, y4 = [gram_matrix(y) for y in style_features]

adam = Adam(lr=args.lr)
model.compile(optimizer=adam,
              loss=[style_reconstruction_loss(y1),
                    style_reconstruction_loss(y2),
                    style_reconstruction_loss(y3),
                    style_reconstruction_loss(y4),
                    feature_reconstruction_loss,
                    total_variation_loss])

"""Dummy arrays
When we use fit(X, y) function,
we must set same shape arrays between X and y.
However, we want to apply array of different shape to the objective function.
So, we prepare for Dummy arrays.
"""
_1 = np.empty((1, 256, 256, 64))
_2 = np.empty((1, 128, 128, 128))
_3 = np.empty((1, 64, 64, 256))
_4 = np.empty((1, 32, 32, 512))
_5 = np.empty((1, 1, 64, 64, 256))
_6 = np.empty((1, 256, 256, 3))

def generate_arrays_from_file():
    while True:
        for path in imagepaths:
            contents_img = load_image(path, image_size)
            yield (contents_img, [_1, _2, _3, _4, _5, _6])

style_name = args.style_image.split("/")[-1].split(".")[0]

print('Num traning images:', nb_data)
print(nb_data, 'iterations,', nb_epoch, 'epochs')
model.fit_generator(generate_arrays_from_file(),
                    samples_per_epoch=nb_data,
                    nb_epoch=nb_epoch)
if not os.path.exists("./weights"):
    os.mkdir("weights")
fsn.save_weights("./weights/{}.hdf5".format(style_name))
print("Saved weights")
