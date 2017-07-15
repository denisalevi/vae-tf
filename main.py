import os
import sys

import numpy as np
import tensorflow as tf

import plot
from utils import get_mnist
import vae


# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_DIM = 28

ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                500, 500, # intermediate encoding
                10] # latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
    "beta": 1.0
}

MAX_ITER = np.inf#2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"
PLOTS_DIR = "./png"


def load_mnist(reshape=False, **kwargs):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data", reshape=reshape, **kwargs)

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
                          outdir=PLOTS_DIR)

def interpolate_digits(model, mnist):
    imgs, labels = mnist.train.next_batch(100)
    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
    mus, _ = model.encode(np.vstack(imgs[i] for i in idxs))
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

    xs = np.squeeze([get_mnist(n, mnist) for n in ns])
    mus, _ = model.encode(xs)
    plot.morph(model, mus, n_per_morph=n_per_morph, outdir=PLOTS_DIR,
               name="morph_{}".format("".join(str(n) for n in ns)))

def main(to_reload=None):
    mnist = load_mnist()

    if to_reload: # restore
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else: # train
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(mnist, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate_every_n=2000,
                verbose=True, save_final_state=True, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=False, plot_subsets_every_n=None, save_summaries_every_n=100)
        v.create_embedding(mnist.test.images, labels=mnist.test.labels, label_names=None,
                           sample_latent=True, latent_space=True, input_space=True,
                           image_dims=(28, 28))

    #all_plots(v, mnist)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--beta', nargs=1, type=float, default=HYPERPARAMS["beta"],
                        help='beta value of betaVAE to use (default: 1.0)')
    parser.add_argument('--arch', nargs='*', type=int, default=ARCHITECTURE[1:],
                        help='decoder/encoder architecture to use')
    args = parser.parse_args()

    tf.reset_default_graph()

    for DIR in (LOG_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    HYPERPARAMS["beta"] = args.beta
    ARCHITECTURE = [IMG_DIM**2] + args.arch
    main()

#    try:
#        to_reload = sys.argv[1]
#        main(to_reload=to_reload)
#    except IndexError:
#        main()
