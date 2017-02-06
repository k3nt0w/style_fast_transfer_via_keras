# style_fast_transfer_via_keras

Implementation of ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155)
 in Keras 1.2.0.

I used this [chainer Implementation](https://arxiv.org/abs/1603.08155) and [Keras exmaple](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py) for as a reference.  

However, I think that the [Keras example](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py) is a little confusing. That's why, I implemented it in a way very different from the example.

I will explain this implementation method on [my blog](http://www.mathgram.xyz) so please refer it.

I hope this implementation method will be of your help.

## Requirement
+ Keras v1.2.0
```
$ pip install keras
```
! caution : I always use tensorflow as backend. So, if you use theano, the program won't work. Then, please change the backend.

## Usage
#### train phase
```
$ python train.py -d <path/to/dataset> -s <path/to/sample image>
```
※ According to the paper, I trained the network using MSCOCO dataset.

#### generate phase

```
$ python generate.py -i <path/to/contents image> -w <path/to/weights of style>
```
example
```
$ python generate.py -i <path/to/contents image> -w <path/to/weights of style>
```

## Example
I trained the network only 2 epochs per style image with [MSCOCO dataset](http://mscoco.org/dataset/#download). If you wanna generate more beautiful image, please increase the number of epoch and retrain the net.
