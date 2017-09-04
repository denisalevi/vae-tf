import random
import argparse

import numpy as np
import tensorflow as tf


def print_(var, name: str, first_n=5, summarize=5):
    """Util for debugging, by printing values of tf.Variable `var` during training"""
    # (tf.Tensor, str, int, int) -> tf.Tensor
    return tf.Print(var, [var], "{}: ".format(name), first_n=first_n,
                    summarize=summarize)

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

def get_deconv_params(out_size, in_size, filter_size, stride):
    """Get parameters for tf.contrib.layer.conv2d_transpose such that the output
    shape equals the input shape of the corresponding conv2d operation.

    The output shape of the conv2d_transpose operation is calculated as:
        for padding == 'SAME':
            out_size = in_size * s + max(filter_size - stride, 0)
        for padding == 'VALID':
            out_size = in_size * s

    If the input size is smaller then the filter size, use 'VALID' padding and
    try to use the same stride as in the conv2d operation and change the
    filter_size. If not possible, decrease the stride.
    If the input size is not smalle then the filter size, try to achieve the desired
    output shape with 'SAME' padding by changing the stride. If not possible, use
    'VALID' padding.

    :param out_size: The desired output shape (input of the conv2d operation)
    :param in_size: The current input shape
    :param filter_size: The filter size used in the conv2d operation
    :param stride: The stride used in the conv2d operation
    """
    out_size = np.asarray(out_size)
    in_size = np.asarray(in_size)
    filter_size = np.asarray(filter_size)
    stride = np.asarray(stride)

    for var in [out_size, in_size, filter_size, stride]:
        #print(var, np.isscalar(var), type(var), var.shape)
        assert len(var) == 2, \
                'All params need to be be of len 2, got {}'.format(var)

    if np.any(in_size < filter_size) or not np.all(np.mod(out_size, in_size) == 0):
        padding = 'VALID'
        # tf.contrib.layer.conv2d_transpose calculates the output shape as
        # out_size = in_size * s + max(filter_size - stride, 0)
        filter_size = out_size - stride * (in_size - 1)
        if np.any(stride > filter_size) and not np.all(stride > filter_size):
            print("WARNING: Changing x/y ratio of filter/stride in deconvolution")
        for n in range(2):
            if stride[n] > filter_size[n]:
                stride[n] = stride[n] - 1
                filter_size[n] = out_size[n] - stride[n] * (in_size[n] - 1)
                assert stride[n] <= filter_size[n], \
                        'Bug in calculating deconvolution output shape. n={} '\
                        'stride[n]={}, filter[n]={}'.format(n, stride[n],
                                                                filter_size[n])
    else:
        padding = 'SAME'
        # tf.contrib.layer.conv2d_transpose calculates the output shape as
        # out_shape = in_shape * stride
        stride = out_size // in_size  # if this is no int, use VALID padding
    return filter_size, stride, padding

def fc_or_conv_arg(string):
    '''
    Checks argparse argument for compatibality with VAE layer specification.

    `string` must be either convertible to integer (FC layer) or consist of
    2 to 4 space seperated values (CONV layer) of which the first 3 must be
    convertible to integer and the 4th (if specified) must be SAME or VALID
    (case insensitive)

    Parameters
    ----------
    string : str
        The string that argparse passes from the commande line

    Returns
    -------
    int or list
        The corresponding list of arguments (except of the input size) that can
        be passed to the architecutre keyword of the VAE initializer.

    Raises
    ------
    argparse.ArgumentTypeError
        If `string` can't be converted into a correct architecture argument.
    '''
    try:
        return int(string)
    except ValueError:
        pass

    try:
        args = string.split(" ")
        assert 2 <= len(args) <= 4
        out = []
        for n, arg in enumerate(args):
            if n == 3:
                # padding argument (str)
                assert arg.lower() in ['same', 'valid']
                out.append(arg)
            else:
                # integer arguments
                out.append(int(arg))
        return tuple(out)
    except:
        pass

    msg = 'Argmuents must be integer (FC layer) or strings of 2 to 4 space '\
          'seperated values (CONV layer) where the first 3 values must be '\
          'integers and the last must be `SAME` or `VALID`. Got `{}`'\
          .format(string)
    raise argparse.ArgumentTypeError(msg)

# Adapted from https://github.com/InFoCusp/tf_cnnvis/blob/master/tf_cnnvis/utils.pyp
def convert_into_grid(Xs, padding=1, grid_dims=None):
    '''
    Convert 4-D numpy array into a grid image

    Parameters
    ----------
    Xs : ndarray
        4D array of images to make grid out of it
    padding : int, optional
        padding size between grid cells
    grid_dims : list, optional
        length 2 list of grid dimensions

    Returns
    -------
    ndarray
        3D array, grid of input images 
    '''
    (N, H, W, C) = Xs.shape
    if grid_dims is None:
        grid_size_h = grid_size_w = int(np.ceil(np.sqrt(N)))
    else:
        assert len(grid_dims) == 2
        grid_size_h, grid_size_w = grid_dims
    grid_height = H * grid_size_h + padding * (grid_size_h - 1)
    grid_width = W * grid_size_w + padding * (grid_size_w - 1)
    grid = np.ones((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size_h):
        x0, x1 = 0, W
        for x in range(grid_size_w):
            if next_idx < N:
                grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid#.astype('uint8')
def _images_to_grid(images):
    """
    Convert a list of arrays of images into a list of grid of images

    :param images: 
        a list of 4-D numpy arrays(each containing images)
    :type images: list

    :return: 
        a list of grids which are grid representation of input images
    :rtype: list
    """
    grid_images = []
    # if 'images' is not empty convert
    # list of images into grid of images
    if len(images) > 0:
        N = len(images)
        H, W, C = images[0][0].shape
        for j in range(len(images[0])):
            tmp = np.zeros((N, H, W, C))
            for i in range(N):
                tmp[i] = images[i][j]
            grid_images.append(np.expand_dims(convert_into_grid(tmp), axis = 0))
    return grid_images
