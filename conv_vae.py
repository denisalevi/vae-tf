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

def build_encoder(self, x, dropout=1, output_size=10):
    '''Create encoder network.
    Args:
    input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
    A tensor that expresses the encoder network
    '''
    encoder = tf.reshape(x, [-1, 28, 28, 1])
    encoder = layers.conv2d(encoder, 32, 5, stride=2)
    encoder = layers.conv2d(encoder, 64, 5, stride=2)
    encoder = layers.conv2d(encoder, 128, 5, stride=2, padding='VALID')
    encoder = layers.dropout(encoder, keep_prob=dropout)
    encoder = layers.flatten(encoder)
    return layers.fully_connected(encoder, output_size, activation_fn=None)

def build_decoder(self, z, dropout=1):
    '''Create decoder network.
    If input tensor is provided then decodes it, otherwise samples from
    a sampled vector.
    Args:
    input_tensor: a batch of vectors to decode
    Returns:
    A tensor that expresses the decoder network
    '''
    decoder = tf.expand_dims(z, 1)
    decoder = tf.expand_dims(decoder, 1)
    decoder = layers.conv2d_transpose(decoder, 128, 3, padding='VALID')
    decoder = layers.conv2d_transpose(decoder, 64, 5, padding='VALID')
    decoder = layers.conv2d_transpose(decoder, 32, 5, stride=2)
    decoder = layers.conv2d_transpose(
        decoder, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
    decoder = layers.flatten(decoder)
    return decoder

VAE._build_decoder = build_decoder
VAE._build_encoder = build_encoder
