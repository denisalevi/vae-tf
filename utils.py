import functools

import numpy as np
from functional import compose, partial
import tensorflow as tf


def composeAll(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)

def print_(var, name: str, first_n=5, summarize=5):
    """Util for debugging, by printing values of tf.Variable `var` during training"""
    # (tf.Tensor, str, int, int) -> tf.Tensor
    return tf.Print(var, [var], "{}: ".format(name), first_n=first_n,
                    summarize=summarize)

def get_mnist(n, mnist):
    """Returns 784-D numpy array for random MNIST digit `n`"""
    assert 0 <= n <= 9, "Must specify digit 0 - 9!"
    import random

    SIZE = 500
    imgs, labels = mnist.train.next_batch(SIZE)
    idxs = iter(random.sample(range(SIZE), SIZE)) # non-in-place shuffle

    for i in idxs:
        if labels[i] == n:
            return imgs[i] # first match

def variable_summaries(variable, scope_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(variable)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', variable)

# Adapted from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        # only one color channel, repeat that 3 times (~ gray scale)
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    # substract min and devide my max for each image (normalise to [0,1])
    # get min value of each image by flattening data along HxWx3
    # --> min.shape == (N,)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    # data.transpose(1,2,3,0).shape == (H,W,3,N) --> broadcasting -min along N
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    # same for max with devision
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0))\
            + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
