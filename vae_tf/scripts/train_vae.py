#!/usr/bin/env python

import os
import sys
import glob
import random

import numpy as np
import tensorflow as tf

from vae_tf import plot
from vae_tf.vae import VAE
from vae_tf.mnist_helpers import load_mnist, get_mnist
from vae_tf.utils import fc_or_conv_arg

# TODO either check if tf_cnnvis is installed or add to package requirements
from tf_cnnvis import activation_visualization, deconv_visualization, deepdream_visualization

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

######################
## MODEL PARAMETERS ##
######################

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
    "beta": 1.0,
    "img_dims": (28, 28)
}

IMG_DIMS = HYPERPARAMS['img_dims']

ARCHITECTURE = [IMG_DIMS[0] * IMG_DIMS[1],  # 784 pixels
                (32, 5, 2, 'SAME'),
                (64, 5, 2, 'SAME'),
                (128, 5, 2, 'SAME'),
                #500, 500,                   # intermediate encoding
                10]                         # latent space dims
                # (and symmetrically back out again)


MAX_ITER = 40000#np.inf#2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"

# visualize convolutional filter activations (output of conv layers) for a given input image
ACTIVATION_VISUALIZATION = True
# visualize conv layer inputs reconstructed from the featuremaps (conv layer outputs) for a given input image
# using deconv operations
DECONV_VISUALIZATION = False

# which MNIST digits to visualize (activation and/or deconv), if None one random is visualized
VISUALIZE_DIGITS = [4] #None # [1, 2, 7]

# where to save cnnvis figures (None does not save on disk)
CNNVIS_OUTDIR = None

######################

def all_plots(model, mnist):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, mnist)

        print("Exploring latent...")
        plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=model.png_dir)
        for n in (24, 30, 60, 100):
            plot.exploreLatent(model, nx=n, ny=n, ppf=True, outdir=model.png_dir,
                               name="explore_ppf{}".format(n))

    print("Interpolating...")
    interpolate_digits(model, mnist)

    print("Plotting end-to-end reconstructions...")
    plot_all_end_to_end(model, mnist)

    print("Morphing...")
    morph_numbers(model, mnist, ns=[9,8,7,6,5,4,3,2,1,0])

    print("Plotting 10 MNIST digits...")
    for i in range(10):
        plot.justMNIST(get_mnist(i, mnist), name=str(i), outdir=model.png_dir)

def plot_all_in_latent(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        plot.plotInLatent(model, dataset.images, dataset.labels, name=name,
                          outdir=outdir)

def interpolate_digits(model, mnist):
    imgs, labels = mnist.train.next_batch(100)
    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
    mus, _ = model.encode(np.stack([imgs[i] for i in idxs], axis=0))
    plot.interpolate(model, *mus, name="interpolate_{}->{}".format(
        *(labels[i] for i in idxs)), outdir=model.png_dir)

def plot_all_end_to_end(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        x, _ = dataset.next_batch(10)
        x_reconstructed = model.vae(x)
        plot.plotSubset(model, x, x_reconstructed, n=10, name=name,
                        outdir=model.png_dir)

def morph_numbers(model, mnist, ns=None, n_per_morph=10):
    if not ns:
        import random
        ns = random.sample(range(10), 10) # non-in-place shuffle

    xs = np.stack([get_mnist(n, mnist) for n in ns])
    mus, _ = model.encode(xs)
    plot.morph(model, mus, n_per_morph=n_per_morph, outdir=model.png_dir,
               name="morph_{}".format("".join(str(n) for n in ns)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--beta', nargs=1, type=float, default=None,
                        help='beta value of betaVAE to use (default: 1.0)')
    parser.add_argument('--arch', nargs='*', type=fc_or_conv_arg, default=None,
                        help='decoder/encoder architecture to use')
    parser.add_argument('--log_folder', default=None,
                        help='where to store the tensorboard summaries')
    parser.add_argument('--reload', type=str, nargs='?', default=None, const='most_recent',
                        help='reload previously trained model (path to meta graph name'\
                             ' without .meta, if none given reloads most recently trained one)')
    parser.add_argument('--plot_all', action='store_true',
                        help='create all plots (latent interpolation, '\
                             'end-to-end reconstruction)')
    parser.add_argument('--no_plots', action='store_true',
                        help="don't plot anything (default plots end-to-end reconstruction)")
    parser.add_argument('--visualize_digits', nargs='+', default=None, type=int, choices=range(0,10),
                        help="pass integers of digits that should be visualized (activation "
                             "and/or deconv, depending on setting in train_vae.py)")
    parser.add_argument('--max_iter', default=None, type=int,
                        help='Number of batches after which to stop training')
    parser.add_argument('--create_embedding', action='store_true',
                        help='Create an embedding of test data for TensorBoard')
    args = parser.parse_args()

    # change model parameters given at start of this file when passed as command-line argument
    if args.log_folder:
        LOG_DIR = 'log_' + args.log_folder

    if args.beta:
        HYPERPARAMS["beta"] = args.beta[0]

    if args.arch:
        ARCHITECTURE = [IMG_DIMS[0] * IMG_DIMS[1]] + args.arch

    if args.visualize_digits:
        VISUALIZE_DIGITS = args.visualize_digits

    if args.max_iter:
        MAX_ITER = args.max_iter

    try:
        os.mkdir(LOG_DIR)
    except FileExistsError:
        pass

    # train or reload model
    mnist = load_mnist()

    tf.reset_default_graph()

    if args.reload:  # restore
        if args.reload == 'most_recent':
            # load most recently trained vae model (assuming it hasen't been first reloaded)
            log_folders = [path for path in glob.glob('./' + LOG_DIR + '/*')
                           if not 'reloaded' in path]
            log_folders.sort(key=os.path.getmtime)
            # load trained VAE from last checkpoint
            assert len(log_folders) > 1, 'log folder is empty, nothing to reload'
            meta_graph = os.path.abspath(tf.train.latest_checkpoint(log_folders[-1]))
        else:
            meta_graph = args.reload

        model = VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=meta_graph)
        print("Loaded model from {}".format(meta_graph))

        # don't plot by default when reloading (--plot_all overwrites this)
        args.no_plots = True
    else:  # train
        model = VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        model.train(mnist, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate_every_n=2000,
                verbose=True, save_final_state=True, plots_outdir=None,
                plot_latent_over_time=False, plot_subsets_every_n=None, save_summaries_every_n=100)
        if args.create_embedding:
            subset_size = 1000
            subset_images, subset_labels = random_subset(mnist.test.images, subset_size,
                                                         labels=mnist.test.labels,
                                                         same_num_labels=True)
            model.create_embedding(subset_images, labels=subset_labels, label_names=None,
                                   sample_latent=False, latent_space=True, input_space=True,
                                   create_sprite=True)
        meta_graph = model.final_checkpoint

    if not all(isinstance(layer, int) for layer in ARCHITECTURE):
        # we have convolutional layers
        conv_layers = []
        deconv_layers = []
        for i in model.sesh.graph.get_operations():
            if i.type.lower() == 'conv2d':# biasadd, elu, relu, conv2dbackpropinput
                if not 'optimizer' in i.name:
                    # don't visualise convolution operation of optimizer operations
                    conv_layers.append(i.name)
                else:
                    print('Skipping filter visualization for {}'.format(i.name))
            elif i.type.lower() == 'conv2dbackpropinput':
                if not 'optimizer' in i.name and not i.name.startswith('decoding_1'):
                    # decoding and decoding_1 are sharing weights
                    deconv_layers.append(i.name)
                else:
                    print('Skipping filter visualization for {}'.format(i.name))
        # tf_cnnvis takes the .meta file as meta_graph input
        meta_graph_file = '.'.join([meta_graph, 'meta'])

        if VISUALIZE_DIGITS is None:
            VISUALIZE_DIGITS = [random.randint(0, 9)]

        for digit in VISUALIZE_DIGITS:
            x_in = np.expand_dims(get_mnist(digit, mnist), 0)

            if ACTIVATION_VISUALIZATION:
                activation_visualization(graph_or_path=meta_graph_file,
                                         value_feed_dict={model.x_in: x_in},
                                         layers=conv_layers+deconv_layers,#['c'],
                                         path_logdir=os.path.join(model.log_dir, 'viz'),
                                         path_outdir=CNNVIS_OUTDIR,
                                         name_suffix=str(digit))
            if DECONV_VISUALIZATION:
                deconv_visualization(graph_or_path=meta_graph_file,
                                     value_feed_dict={model.x_in: x_in},
                                     layers=conv_layers,#['c'],
                                     path_logdir=os.path.join(model.log_dir, 'viz'),
                                     path_outdir=CNNVIS_OUTDIR,
                                     name_suffix=str(digit))

    # default plot=False, no_plot=False --> plot_all_end_to_end()
    if args.plot_all:
        all_plots(model, mnist)
    elif not args.no_plots:
        plot_all_end_to_end(model, mnist)
