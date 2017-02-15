from model import *
from preprocess import *
from loss import *
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import argparse
import os
import sys

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
parser.add_argument('--batchsize', '-b', default=1, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--bound', '-bo', default=0, type=int)

args = parser.parse_args()

image_size = args.image_size
nb_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style

""" "imagepaths" is a list containing absolute paths of Dataset.
"""
fs = os.listdir(args.dataset)
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)

if args.bound:
    imagepaths = imagepaths[:args.bound]
nb_data = len(imagepaths)

style = np.asarray(Image.open(args.style_image).convert('RGB').resize((image_size,image_size)), dtype=np.float32)
style = np.expand_dims(style, axis=0)
y1, y2, y3, y4 = get_style_features(style, image_size, image_size)

fsn = FastStyleNet(train_flag=True)
model = fsn.create_model()

#if len(args.weight) > 0:
#    model.load_weights(args.weight)

adam = Adam(lr=args.lr)
model.compile(optimizer=adam,
              loss=[style_reconstruction_loss(y1),
                    style_reconstruction_loss(y2),
                    style_reconstruction_loss(y3),
                    style_reconstruction_loss(y4),
                    feature_reconstruction_loss,
                    total_variation_loss],
               loss_weights=[lambda_s, lambda_s, lambda_s,
                             lambda_s, lambda_f, lambda_tv])

"""Dummy arrays
When we use fit(X, y) function,
we must set same shape arrays between X and y.
However, we want to apply array of different shape to the objective function.
So, we prepare for Dummy arrays.
"""
_1 = np.empty((1, 256, 256,   3))
_2 = np.empty((1, 128, 128, 128))
_3 = np.empty((1,  64,  64, 256))
_4 = np.empty((1,  32,  32, 512))
_5 = np.empty((1,  64,  64, 256))
_6 = np.empty((1, 256, 256,   3))

def generate_arrays_from_file():
    while True:
        for path in imagepaths:
            contents_img = load_image(path, image_size)
            yield ([contents_img, contents_img.copy()],
                   [_1, _2, _3, _4, _5, _6])

style_name = args.style_image.split("/")[-1].split(".")[0]

print('Num traning images:', nb_data)
print(nb_data, 'iterations,', nb_epoch, 'epochs')

model.fit_generator(generate_arrays_from_file(),
                    samples_per_epoch=nb_data,
                    nb_epoch=nb_epoch)

fsn.save_fastnet_weights(style_name, directory="weights/")
