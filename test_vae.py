import os
import sys
import shutil

import numpy as np
import tensorflow as tf
from nose.tools import with_setup

import plot
from utils import get_mnist
import vae

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

MAX_ITER = np.inf
MAX_EPOCHS = np.inf

LOG_DIR = "/tmp/test_log"
PLOTS_DIR = "/tmp/test_plots"

def load_mnist(**kwargs):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data", **kwargs)

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
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
    v.train(mnist, max_iter=2, max_epochs=1, cross_validate_every_n=1,
            verbose=False, save_final_state=True, plots_outdir=PLOTS_DIR,
            plot_latent_over_time=False, plot_subsets_every_n=1,
            save_summaries_every_n=1, save_input_embedding=True,
            save_latent_embedding=True)

@with_setup(clear_dirs)
def test_reloading_meta_graph():
    clear_dirs()
    tf.reset_default_graph()
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
    checkpoint = v.save_final_checkpoint()
    tf.reset_default_graph()
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=checkpoint)
    _ = v.save_final_checkpoint()
    tf.reset_default_graph()
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=checkpoint)
    checkpoint = v.save_final_checkpoint()
    tf.reset_default_graph()
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=checkpoint)

if __name__ == '__main__':
    test_reloading_meta_graph()
