from model import *
from preprocess import *

import argparse

parser = argparse.ArgumentParser(description='Real-time style transfer via Keras')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--epoch', '-e', default=2, type=int)

fsn = FastStyleNet()
fsn.compile(optimizer="adam", loss="mse")

args = parser.parse_args()

fs = os.listdir(args.dataset)
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)

nb_data = len(imagepaths)

def generate_arrays_from_file():
    while True:
        for path in imagepaths:
            contents_img = load_image(path, 256)
            yield (contents_img, contents_img)

fsn.fit_generator(generate_arrays_from_file(),
                    samples_per_epoch=nb_data,
                    nb_epoch=args.epoch)

fsn.save_weights("./weights/pretrained.hdf5")
