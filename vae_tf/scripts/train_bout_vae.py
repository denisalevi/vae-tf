#!/usr/bin/env python

import os
import sys
import glob
import random
import time

import numpy as np
import tensorflow as tf

from vae_tf import plot
from vae_tf.vae import VAE
from vae_tf.bout_helpers.datasets import load_bouts, get_bout
from vae_tf.utils import fc_or_conv_arg, random_subset

# TODO either check if tf_cnnvis is installed or add to package requirements
from tf_cnnvis import activation_visualization, deconv_visualization, deepdream_visualization

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#DATADIR = '/home/denisalevi/projects/deep_learning_champalimaud/data/bout_data/tfrecords_standard_all_data_scaled_01/bout_data.npz'
#DATADIR = '/home/denisalevi/projects/deep_learning_champalimaud/data/bout_data/tfrecords_standard_all_data_scaled_01_with_bout_npz/bout_data.npz'
DATADIR = '/home/denisalevi/projects/deep_learning_champalimaud/data/bout_data/tfrecords_one_exp/bout_data.npz'

######################
## MODEL PARAMETERS ##
######################

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 2E-5,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.relu,
    "squashing": tf.nn.relu,#sigmoid,
    "beta": 1.0,
    "img_dims": None,  # set this after loading bout data
    "data_range": (0, 1),#(np.iinfo(np.int16).min, np.iinfo(np.int16).max),
    #"reconstruction_loss": "l2_loss"
    #"reconstruction_loss": "sqrerr"
    #"reconstruction_loss": "lsd"
    #"reconstruction_loss": "crossEntropy"
    "reconstruction_loss": "crossEntropy2"
}

ARCHITECTURE = [None,  # will be set in __name__ == '__main__' part
                (64, (8,300), (1,1), 'SAME'),
                (64, (1,10), (1,2), 'SAME'),
                (128, (2,10), (1,2), 'SAME'),
                (256, (4,10), (2,2), 'SAME'),
                (512, (4,10), (2,2), 'SAME'),
                #500, 500,                   # intermediate encoding
                50]                         # latent space dims
                # (and symmetrically back out again)


MAX_ITER = 20000#np.inf#2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"

# visualize convolutional filter activations (output of conv layers) for a given input image
ACTIVATION_VISUALIZATION = True
# visualize conv layer inputs reconstructed from the featuremaps (conv layer outputs) for a given input image
# using deconv operations
DECONV_VISUALIZATION = False

# which MNIST digits to visualize (activation and/or deconv), if None one random is visualized
VISUALIZE_LABELS = [4] #None # [1, 2, 7]
VISUALIZE_LABEL_TYPE = 'marques'

# where to save cnnvis figures (None does not save on disk)
CNNVIS_OUTDIR = None

######################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--beta', nargs=1, type=float, default=None,
                        help='beta value of betaVAE to use (default: 1.0)')
    parser.add_argument('--arch', nargs='*', type=fc_or_conv_arg, default=None,
                        help='decoder/encoder architecture to use')
    parser.add_argument('--data', default=None, type=str,
                        help='`.npz` data archive with bout data')
    parser.add_argument('--log_folder', default=None,
                        help='where to store the tensorboard summaries')
    parser.add_argument('--reload', type=str, nargs='?', default=None, const='most_recent',
                        help='reload previously trained model (path to meta graph name'\
                             ' without .meta, if none given reloads most recently trained one)')
    parser.add_argument('--no_plots', action='store_true',
                        help="don't plot anything (default plots end-to-end reconstruction)")
    parser.add_argument('--visualize_labels', nargs='+', default=None, type=int,
                        help="pass integers of digits that should be visualized (activation "
                             "and/or deconv, depending on setting in train_vae.py)")
    parser.add_argument('--visualize_label_type', default=None, choices=['marques', 'experiment'],
                        help="Choses the label type for activation visualisation of specified "
                             "label IDs (from VISUALIZE_LABELS or args.visualize_labels)")
    parser.add_argument('--max_iter', default=None, type=int,
                        help='Number of batches after which to stop training')
    parser.add_argument('--create_embedding', action='store_true',
                        help='Create an embedding of test data for TensorBoard')
    parser.add_argument('--save_pngs', nargs='?', default=None, type=str, const=True, 
                        help='Save figures as pngs. Optionally pass target folder as argument.')
    parser.add_argument('--test', action='store_true',
                        help="Don't load dataset, instead just create random batch and stop "
                             "after first training batch. For code testing.")
    parser.add_argument('--experiments', nargs='+', default='all', type=str,
                        help='Names of experiments to load.')
    parser.add_argument('--in_dims', nargs=2, default=[8, 300], type=int,
                        help='number of tail fragments, number of frames to crop / pad data to')
    parser.add_argument('--params', nargs='+', default=None, type=lambda kv: kv.split("="),
                        help='hyperparams for vae')
    parser.add_argument('--latent', default=10, type=int,
                        help='size of latent space')
    parser.add_argument('--cont', nargs='?', default=None, type=str, const='most_recent',
                        help='path to metagraph (without .meta)')
    args = parser.parse_args()

    if args.test:
        DATADIR = '/home/denisalevi/projects/deep_learning_champalimaud/data/minExampleBoutFiles/boutFilesFinal_npz/one_exp_padded_300/bout_data.npz'
    elif args.data is not None:
        DATADIR = args.data

    # set VAE.img_dims from bout data
    #IMG_DIMS = bout_data.train.bout_dims
    IMG_DIMS = (args.in_dims)
    HYPERPARAMS['img_dims'] = IMG_DIMS
    ARCHITECTURE[0] = IMG_DIMS[0] * IMG_DIMS[1]

    if args.latent:
        ARCHITECTURE[-1] = args.latent

    hyper_kwargs = {}
    if args.params:
        for key, value in args.params:
            # if value is numeric, turn into float
            try:
                value = float(value)
            except ValueError:
                pass
            hyper_kwargs[key] = value

    HYPERPARAMS.update(hyper_kwargs)

    # change model parameters given at start of this file when passed as command-line argument
    if args.log_folder:
        LOG_DIR = 'log_' + args.log_folder

    if args.beta:
        HYPERPARAMS["beta"] = args.beta[0]

    if args.arch:
        ARCHITECTURE = [IMG_DIMS[0] * IMG_DIMS[1]] + args.arch

    if args.visualize_labels:
        VISUALIZE_LABELS = args.visualize_labels

    if args.visualize_label_type:
        VISUALIZE_LABEL_TYPE = args.visualize_label_type

    if args.test:
        MAX_ITER = 1

    if args.max_iter:
        MAX_ITER = args.max_iter

    try:
        os.mkdir(LOG_DIR)
    except FileExistsError:
        pass

    tf.reset_default_graph()

    # train or reload model
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

        # don't plot by default when reloading
        args.no_plots = True
    if not args.reload:  # train
        if args.cont is not None:  # continue from reloaded model
            if args.cont == 'most_recent':
                # load most recently trained vae model (assuming it hasen't been first reloaded)
                log_folders = [path for path in glob.glob('./' + LOG_DIR + '/*')
                               if not 'reloaded' in path]
                log_folders.sort(key=os.path.getmtime)
                # load trained VAE from last checkpoint
                assert len(log_folders) > 1, 'log folder is empty, nothing to reload'
                meta_graph = os.path.abspath(tf.train.latest_checkpoint(log_folders[-1]))
            else:
                meta_graph = args.cont

            model = VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=meta_graph)
            print("Loaded model from {}".format(meta_graph))
        else:  # new model
            model = VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)

        # seems that the dataset handles need to created after VAE init (session creation) and before training

        bout_data = load_bouts(DATADIR, read_threads=8, tfrecords=True)#, experiments=args.experiments)#['3minLightDark2'])
        scale, shift = bout_data['to_rad_transform']
        # with Z = scale * X + shift (transormation into radian representation)
        # the zero radian (Z = 0) was transformed to X = - shift / scale
        zero_radian = - shift / scale
        for name in ['train', 'test', 'validation']:
            dataset = bout_data[name]
            dataset = dataset.padded_batch(model.batch_size,
                                           padded_shapes=((*model.img_dims, 1), (), ()),
                                           padding_values=(tf.cast(zero_radian, tf.float32), -1, -1))
            bout_data[name] = dataset

        if not args.test and (ACTIVATION_VISUALIZATION or DECONV_VISUALIZATION):
            digit_iterator_handles = []
            for label_id in VISUALIZE_LABELS:
                if VISUALIZE_LABEL_TYPE == 'marques':
                    label_type = 'marques_classification'
                elif VISUALIZE_LABEL_TYPE == 'experiment':
                    label_type = 'experiment_id'
                        
                iterator_handle = get_bout(label_id, bout_data, label_type, dataset='test',
                                           return_handle=True)
                digit_iterator_handles.append(model.sesh.run(iterator_handle))

        model.train(bout_data['train'], max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                    cross_validate_every_n=2000, validation_dataset=bout_data['validation'],
                    verbose=True, save_final_state=True, plots_outdir=None,
                    plot_latent_over_time=False, plot_subsets_every_n=None, save_summaries_every_n=100)
        if args.create_embedding:
            # TODO need fixing, maybe Dataset.read_batch_features or sesh.run a certain batch size or Dataset.filter
            # or Dataset.rejection_sampling or just precompute a sprite immage for embeddings...
            subset_size = 100
            # TODO use label names instead of IDs here
            labels = np.vstack([bout_data.test.marques_labels, bout_data.test.experiment_ids]).T
            label_names = ['marques_classification', 'experiment_id']
            subset_images, subset_labels = random_subset(bout_data.test.bouts, subset_size,
                                                         labels=labels,
                                                         same_num_labels=True)

            def sprite_function(bouts):
                bout_len = bout_data.train.bouts.shape[2]
                int16_to_rad_factor = bout_data.train.int16_to_rad_factor
                # sprite will be 30x30=900 images with 273*30=8190 pixel per dimension
                max_px = 273
                return plot.convert_bout_data_to_rgb_plots(bouts, bout_len, int16_to_rad_factor,
                                                           max_px)

            model.create_embedding(subset_images, labels=subset_labels, label_names=label_names,
                                   sample_latent=False, latent_space=True, input_space=True,
                                   create_sprite=sprite_function,
                                   invert_sprite=False)
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

        if VISUALIZE_LABELS is None:
            VISUALIZE_LABELS = [random.randint(0, 9)]

        #if not args.test:
        #    for label_id, handle in zip(VISUALIZE_LABELS, digit_iterator_handles):

        #        if ACTIVATION_VISUALIZATION:
        #            activation_visualization(graph_or_path=meta_graph_file,
        #                                     value_feed_dict={model.dataset_in: iterator_handle},
        #                                     layers=conv_layers+deconv_layers,#['c'],
        #                                     path_logdir=model.validation_writer_dir,
        #                                     path_outdir=CNNVIS_OUTDIR,
        #                                     # TODO use label name here
        #                                     name_suffix=str(label_id))
        #        if DECONV_VISUALIZATION:
        #            deconv_visualization(graph_or_path=meta_graph_file,
        #                                 value_feed_dict={model.dataset_in: iterator_handle},
        #                                 layers=conv_layers,#['c'],
        #                                 path_logdir=model.validation_writer_dir,
        #                                 path_outdir=CNNVIS_OUTDIR,
        #                                 # TODO use label name here
        #                                 name_suffix=str(label_id))

    # default plot=False, no_plot=False --> plot.plot_reconstructions()
    if not args.no_plots:
        print('SCALE', scale, 'SHIFT', shift)
        transform_kwargs = {'to_rad_transform' : (scale, shift),
                            'bout_len' : model.img_dims[1],
                            'ylim' : 1.7 / np.pi * np.array([-1, 1])}#(-1, 1)}
        start = time.time()
        plot.plot_reconstructions(model, bout_data, n=3, tf_summary=True, save_png=args.save_pngs,
                                  transform_data=plot.plot_bout_as_rgb,
                                  transform_kwargs=transform_kwargs)
        print('plotting reconstruction took {}s'.format(time.time() - start))
        start = time.time()
        plot.plot_reconstructions(model, bout_data, n=3, tf_summary=True, save_png=args.save_pngs,
                                  transform_data=plot.plot_bout_as_rgb,
                                  transform_kwargs=transform_kwargs)
        print('plotting reconstruction took {}s'.format(time.time() - start))
