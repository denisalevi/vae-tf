import os
import sys
import glob

import numpy as np
import tensorflow as tf

from vae_tf import plot
from vae_tf.vae import VAE
from vae_tf.mnist_helpers import load_mnist, get_mnist


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
                500, 500,                   # intermediate encoding
                10]                         # latent space dims
                # (and symmetrically back out again)


MAX_ITER = 40000#np.inf#2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"
PLOTS_DIR = "./png"

######################

def all_plots(model, mnist):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, mnist)

        print("Exploring latent...")
        plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=PLOTS_DIR)
        for n in (24, 30, 60, 100):
            plot.exploreLatent(model, nx=n, ny=n, ppf=True, outdir=PLOTS_DIR,
                               name="explore_ppf{}".format(n))

    print("Interpolating...")
    interpolate_digits(model, mnist)

    print("Plotting end-to-end reconstructions...")
    plot_all_end_to_end(model, mnist)

    print("Morphing...")
    morph_numbers(model, mnist, ns=[9,8,7,6,5,4,3,2,1,0])

    print("Plotting 10 MNIST digits...")
    for i in range(10):
        plot.justMNIST(get_mnist(i, mnist), name=str(i), outdir=PLOTS_DIR)

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
        *(labels[i] for i in idxs)), outdir=PLOTS_DIR)

def plot_all_end_to_end(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        x, _ = dataset.next_batch(10)
        x_reconstructed = model.vae(x)
        plot.plotSubset(model, x, x_reconstructed, n=10, name=name,
                        outdir=PLOTS_DIR)

def morph_numbers(model, mnist, ns=None, n_per_morph=10):
    if not ns:
        import random
        ns = random.sample(range(10), 10) # non-in-place shuffle

    xs = np.stack([get_mnist(n, mnist) for n in ns])
    mus, _ = model.encode(xs)
    plot.morph(model, mus, n_per_morph=n_per_morph, outdir=PLOTS_DIR,
               name="morph_{}".format("".join(str(n) for n in ns)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--beta', nargs=1, type=float, default=None,
                        help='beta value of betaVAE to use (default: 1.0)')
    parser.add_argument('--arch', nargs='*', type=int, default=None,
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
    args = parser.parse_args()

    tf.reset_default_graph()

    # change model parameters given at start of this file when passed as command-line argument
    if args.log_folder:
        LOG_DIR = 'log_' + args.log_folder
        # TODO this should go where embedding goes etc! no extra folder
        PLOTS_DIR = 'png_' + args.log_folder

    for DIR in (LOG_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    if args.beta:
        HYPERPARAMS["beta"] = args.beta[0]
    if args.arch:
        ARCHITECTURE = [IMG_DIMS[0] * IMG_DIMS[1]] + args.arch

    # train or reload model
    mnist = load_mnist()

    if args.reload:  # restore
        if args.reload == 'most_recent':
            # load most recently trained vae model (assuming it hasen't been first reloaded)
            log_folders = [path for path in glob.glob('./' + LOG_DIR + '/*')
                           if not 'reloaded' in path]
            log_folders.sort(key=os.path.getmtime)
            # load trained VAE from last checkpoint
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
                verbose=True, save_final_state=True, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=False, plot_subsets_every_n=None, save_summaries_every_n=100)
        model.create_embedding(mnist.test.images, labels=mnist.test.labels, label_names=None,
                           sample_latent=True, latent_space=True, input_space=True,
                           image_dims=(28, 28))

    # default plot=False, no_plot=False --> plot_all_end_to_end()
    if args.plot_all:
        all_plots(model, mnist)
    elif not args.no_plots:
        plot_all_end_to_end(model, mnist)
