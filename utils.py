import functools
import random

import numpy as np
from functional import compose, partial
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet


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

    SIZE = 500
    imgs, labels = mnist.train.next_batch(SIZE)
    idxs = iter(random.sample(range(SIZE), SIZE)) # non-in-place shuffle

    for i in idxs:
        if labels[i] == n:
            return imgs[i] # first match

def variable_summaries(variable, scope_name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name, default_name='{}_summaries'.format(variable.op.name)):
        mean = tf.reduce_mean(variable)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', variable)

# Adapted from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data, invert=False):
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
    if invert:
        data = 1 - data

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

def random_subset(images, size, labels=None, same_num_labels=False):
    """random_subset returns a shuffled random subset of size size from dataset

    :param dataset: tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object
    :param size: int, size of subset
    :param same_num_labels: bool, weather to return equal number of sampels per label
    """
    assert labels is not None or not same_num_labels, 'no labels given for same_num_labels==True'
    assert images.ndim == 2, 'datasets.images need to be reshaped to (N, W*H)'
    new_labels = []
    if same_num_labels:
        if labels.ndim == 1 or labels.shape[1] == 1:
            split_labels = labels.flat
        else:
            split_labels = labels[:, 0]
        unique_labels = np.unique(split_labels)
        num_labels = unique_labels.size
        subset_per_label = size // unique_labels.size
        remainder = size - subset_per_label * unique_labels.size
        new_images = []
        for n, label in enumerate(unique_labels):
            label_images = images[split_labels == label]
            label_size = subset_per_label + 1 if n < remainder else subset_per_label
            perm = np.arange(label_images.shape[0])
            np.random.shuffle(perm)
            label_subset = perm[:label_size]
            new_images.append(label_images[label_subset])
            #new_labels.append([label] * label_size)
            new_label = labels[split_labels == label]
            new_labels.append(new_label[label_subset])
        perm = np.arange(size)
        np.random.shuffle(perm)
        new_images = np.concatenate(new_images)[perm]
        new_labels = np.concatenate(new_labels)[perm]
    else:
        perm = np.arange(images.shape[0])
        np.random.shuffle(perm)
        subset = perm[:size]
        new_images = images[subset]
        if labels is not None:
            new_labels = labels[subset]
    return new_images, new_labels
