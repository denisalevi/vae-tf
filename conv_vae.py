from vae import VAE

from datetime import datetime
import os
import re
import sys
import scipy

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import layers

import plot
from utils import composeAll, print_, images_to_sprite
from vae import VAE
import numpy as np

def build_encoder(self, x, dropout=1, output_size=10):
    '''Create encoder network.
    Args:
    input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
    A tensor that expresses the encoder network
    '''
    o = []
    encoder = tf.reshape(x, [-1, 28, 28, 1])
    o.append(np.array(encoder.get_shape().as_list()[1]))
    print('in', encoder.get_shape())
    encoder = layers.conv2d(encoder, 32, 5, stride=2, padding='SAME')
    o.append(np.array(encoder.get_shape().as_list()[1]))
    print('32x5x5-2x2-S', encoder.get_shape())
    encoder = layers.conv2d(encoder, 64, 5, stride=2, padding='SAME')
    o.append(np.array(encoder.get_shape().as_list()[1]))
    print('64x5x5-2x2-S', encoder.get_shape())
    encoder = layers.conv2d(encoder, 128, 5, stride=2, padding='VALID')
    o.append(np.array(encoder.get_shape().as_list()[1]))
    print('128x5x5-2x2-V', encoder.get_shape())
    encoder = layers.dropout(encoder, keep_prob=dropout)
    encoder = layers.flatten(encoder)
    self.shapes = o
    return encoder
    #return layers.fully_connected(encoder, output_size, activation_fn=None)

def build_decoder(self, z, dropout=1):
    '''Create decoder network.
    If input tensor is provided then decodes it, otherwise samples from
    a sampled vector.
    Args:
    input_tensor: a batch of vectors to decode
    Returns:
    A tensor that expresses the decoder network
    '''
#    decoder = tf.expand_dims(z, 1)
#    decoder = tf.expand_dims(decoder, 1)
    stride = 2
    idx = 1
    decoder = tf.reshape(z, [-1, 1, 1, 10])
    print('decoding', decoder.get_shape())
    o = self.shapes[-idx]
    idx += 1
    i = decoder.get_shape().as_list()[1]
    k, s, p = get_k(o, stride, i, 5)
    print('o={}, s={}, i={}, k={}'.format(o, s, i, k))
    decoder = layers.conv2d_transpose(decoder, 128, k, stride=s, padding=p)
    print('128x{k}x{k}-{s}x{s}-{p}'.format(k=k, s=s, p=p), decoder.get_shape())
    decoder = layers.fully_connected(decoder, 128*k*k, activation_fn=None)
    print('after fc'.format(k=k, s=s, p=p), decoder.get_shape())
    #k, s = get_k(self.shapes.pop(), s, decoder.get_shape().as_list()[1])
    o = self.shapes[-idx]
    idx += 1
    i = decoder.get_shape().as_list()[1]
    k, s, p = get_k(o, stride, i, 5)
    print('o={}, s={}, i={}, k={}'.format(o, s, i, k))
    decoder = layers.conv2d_transpose(decoder, 64, k, stride=s, padding=p)
    print('64x{k}x{k}-{s}x{s}-{p}'.format(k=k, s=s, p=p), decoder.get_shape())
    #k, s = get_k(self.shapes.pop(), s, decoder.get_shape().as_list()[1])
    o = self.shapes[-idx]
    idx += 1
    i = decoder.get_shape().as_list()[1]
    k, s, p = get_k(o, stride, i, 5)
    print('o={}, s={}, i={}, k={}'.format(o, s, i, k))
    decoder = layers.conv2d_transpose(decoder, 32, k, stride=s, padding=p)
    print('32x{k}x{k}-{s}x{s}-{p}'.format(k=k, s=s, p=p), decoder.get_shape())
    #k, s = get_k(self.shapes.pop(), s, decoder.get_shape().as_list()[1])
    print('o={}, s={}, i={}, k={}'.format(o, s, i, k))
    o = self.shapes[-idx]
    idx += 1
    i = decoder.get_shape().as_list()[1]
    k, s, p = get_k(o, stride, i, 5)
    decoder = layers.conv2d_transpose(decoder, 1, k, stride=s, activation_fn=tf.nn.sigmoid, padding=p)
    print('1x{k}x{k}-{s}x{s}-{p}'.format(k=k, s=s, p=p), decoder.get_shape())
    decoder = layers.flatten(decoder)
    return decoder

VAE._build_decoder = build_decoder
VAE._build_encoder = build_encoder

def get_k(o, s, i, k):
    #return get_deconv_params(o, i, k, s)
    if i < k or o % i != 0:
        #s_ = s - 1
        p = 'VALID'
        k = int(o - s * (i - 1))
        if s > k:
            s = s - 1
            k = int(o - s * (i - 1))
            assert s <= k
    else:
        p = 'SAME'
        s = o // i
    return int(k), int(s), p

def get_deconv_params(out_size, in_size, filter_size, stride):

    for var in [out_size, in_size, filter_size, stride]:
        print(var, np.isscalar(var), type(var), var.shape)
        assert np.isscalar(var) or len(var) == 2, \
                'All params need to be be of len 2, got {}'.format(var)

    out_size = np.asarray(out_size, dtype=int)
    in_size = np.asarray(in_size, dtype=int)
    filter_size = np.asarray(filter_size, dtype=int)
    stride = np.asarray(stride, dtype=int)


    if any(in_size < filter_size) or not all(np.modulo(out_size, in_size) == 0):
        padding = 'VALID'
        # tf.contrib.layer.conv2d_transpose calculates the output shape as
        # out_size = in_size * s + max(filter_size - stride, 0)
        filter_size = out_size - stride * (in_size - 1)
        if stride > filter_size:
            stride = stride - 1
            filter_size = int(out_size - stride * (in_size - 1))
            assert stride <= filter_size, \
                    'Bug in calculating deconvolution output shape'
    else:
        padding = 'SAME'
        # tf.contrib.layer.conv2d_transpose calculates the output shape as
        # out_shape = in_shape * stride
        stride = out_size // in_size  # if this is no int, use VALID padding
    return filter_size, stride, padding
