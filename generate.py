import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

parser = argparse.ArgumentParser(description='style fast transfer: image generator via Keras')
parser.add_argument('input')
parser.add_argument('--weight', '-w', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--padding', default=50, type=int)
args = parser.parse_args()
