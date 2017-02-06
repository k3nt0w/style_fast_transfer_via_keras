from model import *
from preprocess import *
import numpy as np

from keras.optimizers import Adam
from keras import backend as K

import argparse
import os

def gram_matrix(x):
    x = x[0,:,:,:] # (row, col, ch)
    nrow, ncol, nch = K.int_shape(x)
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features)) / (nrow * ncol * nch)
    return gram

def mean_squared_error(y_true, y_pred, ax):
    return K.mean(K.square(y_pred - y_true), axis=ax)

def style_reconstruction_loss(gram_s):
    def loss_function(y_true, y_pred):
        gram_s_hat = gram_matrix(y_pred)
        return lambda_s*mean_squared_error(gram_s, gram_s_hat, ax=-1)
    return loss_function

def feature_reconstruction_loss(y_true, y_pred):
    return lambda_f*mean_squared_error(y_true, y_pred, ax=-1)

def total_variation_loss(y_true, x):
    assert K.ndim(x) == 4
    img_nrows = 256
    img_ncols = 256
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return lambda_tv*K.sum(K.pow(a + b, 1.25))

# train phase
parser = argparse.ArgumentParser(description='Real-time style transfer via Keras')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
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

fs = os.listdir(args.dataset)

# "imagepaths" is a list containing absolute paths.
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)

nb_data = len(imagepaths)

print('num traning images:', nb_data)
print(nb_data, 'iterations,', nb_epoch, 'epochs')

model, fsn = connect_vgg16()

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

# Dummy arrays
_1 = np.empty((1, 256, 256, 64))
_2 = np.empty((1, 128, 128, 128))
_3 = np.empty((1, 64, 64, 256))
_4 = np.empty((1, 32, 32, 512))
_5 = np.empty((1, 64, 64, 256))
_6 = np.empty((1, 256, 256, 3))


print("Now loading contents image feature...")
cis = []
contents_imgs = []

cis_append = cis.append
contents_imgs_append = contents_imgs.append

for path in imagepaths:
    contents_img = load_image(path, image_size)
    contents_imgs_append(contents_img)

    ci = get_contents_features(contents_img)
    ci = K.eval(ci)
    cis_append(ci)
print("Done!, Start training.")
print("-------------------------------------")

ci = get_contents_features(contents_img)
ci = K.eval(ci)

def generate_arrays_from_file(contents_imgs, cis):
    while True:
        for contents_img, ci in zip(contents_imgs, cis):
            yield (contents_img, [_1, _2, _3, _4, ci, _6])

model.fit_generator(generate_arrays_from_file(contents_imgs, cis),
                    samples_per_epoch=nb_data,
                    nb_epoch=nb_epoch)

style_name = args.style_image.split("/")[-1].split(".")[0]

print("Save weights")
if not os.path.exists("./weights"):
    os.mkdir("weights")
fsn.save_weights("./weights/{}.hdf5".format(style_name))
