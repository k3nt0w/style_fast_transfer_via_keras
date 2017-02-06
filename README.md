# style_fast_transfer_via_keras

Implementation of ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155)
 in Keras 1.2.0.

I used this [chainer Implementation](https://arxiv.org/abs/1603.08155) as a very useful reference.  

By the way, I think the example of [neural_style_transfer](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py) is a little confusing. For that reason, I implemented it in a way very different from the example.

I hope this implementation method will be of your help.

##Requirement
+ Keras v1.2.0
```
$ pip install keras
```
! caution : I always use tensorflow as backend. So, if you use theano, this won't work. Then, please change the backend.

## Usage
#### train phase
```
$ python train.py -d <path/to/dataset> -s <path/to/sample image>
```
※ According to the paper, I trained the network using MSCOCO dataset.

#### generate phase

```
$ python train.py -i <path/to/contents image> -w <path/to/weights of style>
```

## Example
I trained the network only 2 epochs with MSCOCO dataset. If you wanna generate more beautiful image, please increase the number of epoch and retrain the net.
