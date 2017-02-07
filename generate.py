from model import *

import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

parser = argparse.ArgumentParser(description='style fast transfer: image generator via Keras')
parser.add_argument('input')
parser.add_argument('--weight', '-w', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--padding', default=50, type=int)
parser.add_argument('--keep_colors', action='store_true')
parser.set_defaults(keep_colors=False)
args = parser.parse_args()

# I referred to "chainer-fast-neuralstyle".
# https://github.com/yusuketomoto/chainer-fast-neuralstyle/blob/master/generate.py


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

fsn = FastStyleNet()
fsn.load_weights(args.weight)

start = time.time()
original = Image.open(args.input).convert('RGB')
image = np.asarray(original, dtype=np.float32)
image = image.reshape((1,) + image.shape)

if args.padding > 0:
	image = np.pad(image, [[0, 0], [0, 0], [args.padding, args.padding], [args.padding, args.padding]], 'symmetric')

out = fsn.predict(image)[0]
med = Image.fromarray(out)
if args.median_filter > 0:
	med = med.filter(ImageFilter.MedianFilter(args.median_filter))
if args.keep_colors:
    med = original_colors(original, med)
print(time.time() - start, 'sec')

med.save(args.out)
