import os
import sys
import glob

import numpy as np
import tensorflow as tf

from vae_tf import plot
from vae_tf.mnist_helpers import get_mnist
#import conv_vae as vae
from vae_tf import vae
from tf_cnnvis import activation_visualization, deconv_visualization, deepdream_visualization


# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_DIM = 28

ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                #(32, 5, 2, 'SAME'),
                #(64, 5, 2, 'SAME'),
                #(128, 5, 2, 'VALID'),
#                (16, 5, 1, 'SAME'),
#                (16, 5, 2, 'SAME'),
#                (32, 5, 1, 'SAME'),
#                (32, 5, 2, 'SAME'),
#                (64, 5, 1, 'SAME'),
#                (64, 5, 2, 'SAME'),
#                (128, 5, 1, 'SAME'),
#                (128, 5, 2, 'SAME'),
                500,500,
                10] # latent space dims
# (and symmetrically back out again)

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

MAX_ITER = 40000#np.inf#2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log_CNN"
PLOTS_DIR = "./png_CNN"


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
    mus, _ = model.encode(np.stack([imgs[i] for i in idxs]))
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

        conv_layers = []
        deconv_layers = []
        #types = {}
        for i in v.sesh.graph.get_operations():
#            if i.type.lower() not in types:
#                types[i.type.lower()] = i.name
            if i.type.lower() == 'conv2d':# biasadd, elu, relu, conv2dbackpropinput
                if 'optimizer' in i.name:
                    print('Skipping filter visualization for {}'.format(i.name))
                conv_layers.append(i.name)
            elif i.type.lower() == 'conv2dbackpropinput':
                if 'optimizer' in i.name:
                    print('Skipping filter visualization for {}'.format(i.name))
                elif not i.name.startswith('decoding_1'):
                    deconv_layers.append(i.name)

#        print('TYPES')
#        for k, va in types.items():
#            print(k, '\t', va)

        meta_graph = '.'.join([v.final_checkpoint, 'meta'])

#        activation_visualization(graph_or_path=meta_graph,
#                                 value_feed_dict={v.x_in: mnist.test.images[:1]},
#                                 layers=conv_layers + deconv_layers,#['c'],
#                                 path_logdir=os.path.join(v.log_dir, 'viz'))
#                                 #path_outdir=v.log_dir)
#        deconv_visualization(graph_or_path=meta_graph,
#                             value_feed_dict={v.x_in: mnist.test.images[:1]},
#                             layers=conv_layers,#['c'],
#                             path_logdir=os.path.join(v.log_dir, 'viz'))
#                             #path_outdir=v.log_dir)
#        deepdream_visualization(graph_or_path=meta_graph,
#                                value_feed_dict={v.x_in: mnist.test.images[:1]},
#                                layer='encoding/Conv_3/convolution',
#                                classes=[1, 2, 3, 4, 5],
#                                path_logdir=os.path.join(v.log_dir, 'viz'))


    #all_plots(v, mnist)
    plot_all_end_to_end(v, mnist)


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

    #print(sys.argv[1])
    #beta = float(sys.argv[1])
    #HYPERPARAMS["beta"] = beta
    #main()
    #arch = np.array(sys.argv[1:], dtype=np.int)
    #ARCHITECTURE = [IMG_DIM**2] + list(arch)
    #main()
