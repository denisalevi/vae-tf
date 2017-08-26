import os
import sys
import shutil

import numpy as np
import tensorflow as tf
from nose.tools import with_setup

from vae_tf import plot
from vae_tf.utils import get_mnist
from vae_tf.vae import VAE

# turn off tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_DIM = 28

ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                500, 500, # intermediate encoding
                10] # latent space dims
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

LOG_DIR = "/tmp/test_log"
PLOTS_DIR = "/tmp/test_plots"

def load_mnist(reshape=False, **kwargs):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data", reshape=reshape, **kwargs)

def create_dirs():
    for DIR in (LOG_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

def clear_dirs():
    for DIR in (LOG_DIR, PLOTS_DIR):
        try:
            shutil.rmtree(DIR)
        except FileNotFoundError:
            pass

@with_setup(create_dirs, clear_dirs)
def test_mnist_example():
    tf.reset_default_graph()
    mnist = load_mnist()
    v = VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
    v.train(mnist, max_iter=2, max_epochs=1, cross_validate_every_n=1,
            verbose=False, save_final_state=True, plots_outdir=PLOTS_DIR,
            plot_latent_over_time=False, plot_subsets_every_n=1,
            save_summaries_every_n=1)
    v.create_embedding(mnist.train.images, labels=mnist.train.labels,
                       image_dims=(28, 28))

@with_setup(clear_dirs)
def test_reloading_meta_graph():
    clear_dirs()
    tf.reset_default_graph()
    v = VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
    v.save_checkpoint()
    checkpoint = v.final_checkpoint
    tf.reset_default_graph()
    v = VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=checkpoint)
    v.save_checkpoint()
    tf.reset_default_graph()
    v = VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=checkpoint)
    v.save_checkpoint()
    checkpoint = v.final_checkpoint
    tf.reset_default_graph()
    v = VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=checkpoint)

@with_setup(clear_dirs)
def test_architecture_model_naming_and_reloading():
    arch = [
        784,  # if removed, remove [1:] in loop below
        [5, 5, 1, 'SAME'],
        [5, 5, 1, 'same'],
        [5, (5,5), (1,1), 'SAME'],
        [5, 5, 1],
        [5, (5,5), (1,1)],
        [5, 5, 'SAME'],
        [5, 5],
        [5, (5,5)],
        10  # if removed, remove [1:] in loop below
    ]
    myvae = VAE(arch, init=False)
    model_name, *_ = myvae.get_new_layer_architecture(arch)
    split = model_name.split('_')
    for name in split[1:-1]:
        assert name == 'conv-5x5x5-1x1-S', 'unexpected name {}'.format(name)
    assert split[0] == '784', split[0]
    assert split[-1] == '10', split[-1]

    arch_reload = myvae.get_architecture_from_model_name(model_name)
    for layer in arch_reload[1:-1]:
        assert layer[0] == 5, layer[0]
        assert layer[1] == (5, 5), layer[1]
        assert layer[2] == (1, 1), layer[2]
        assert layer[3] == 'SAME', layer[3]
    assert arch_reload[0] == 784, arch_reload[0]
    assert arch_reload[-1] == 10, arch_reload[-1]

    arch2 = [784, 500, 10]
    model_name2, *_ = myvae.get_new_layer_architecture(arch2)
    assert model_name2 == '784_fc-500_10', model_name2
    arch_reload2 = myvae.get_architecture_from_model_name(model_name2)
    assert arch_reload2[1] == 500

if __name__ == '__main__':
    test_architecture_model_naming()
