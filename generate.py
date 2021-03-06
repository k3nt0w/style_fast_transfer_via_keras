from model import *
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

"""I referred to "chainer-fast-neuralstyle".
https://github.com/yusuketomoto/chainer-fast-neuralstyle/blob/master/generate.py
"""

parser = argparse.ArgumentParser(description='style fast transfer: image generator via Keras')
parser.add_argument('input')
parser.add_argument('--weight', '-w', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='./sample_imgs/out.jpg', type=str)
parser.add_argument('--test', '-t', default=False, type=bool)
parser.add_argument('--keep_colors', action='store_true')
parser.set_defaults(keep_colors=False)
args = parser.parse_args()

"""from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
"""
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

original = Image.open(args.input).convert('RGB')
image = np.asarray(original, dtype=np.float32) / 255

row, col = image.shape[:2]

fsn = FastStyleNet(img_height=row, img_width=col, train_flag=True)
model = fsn.create_model(train=False)
model.load_weights(args.weight)

print("Load weights.")

print("Now transforming...")
start = time.time()
image = image.reshape((1,) + image.shape)
result = model.predict(image)
result = np.uint8(result[0])
out = Image.fromarray(result)
if args.keep_colors:
    med = original_colors(original, out)
print(time.time() - start, 'sec')
out.save(args.out)
