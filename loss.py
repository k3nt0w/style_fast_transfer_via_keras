from keras import backend as K
from keras.regularizers import Regularizer

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

def style_reconstruction_loss(ys):
    gram_s = K.variable(ys)
    def loss_function(y_true, y_pred):
        gram_s_hat = gram_matrix(y_pred)
        return K.sum(K.square(gram_s - gram_s_hat))
    return loss_function

def feature_reconstruction_loss(y_true, y_pred):
    """This function will receive a tensor that
    already calculated the square error.
    So, just calculate the average here.
    """
    batch, nrow, ncol, nch = K.int_shape(y_pred)

    return K.sum(y_pred) / (nrow * ncol * nch)

def total_variation_loss(y_true, x):
    """I reffered to
    "https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py"
    """
    assert K.ndim(x) == 4
    img_nrows = 256
    img_ncols = 256
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
