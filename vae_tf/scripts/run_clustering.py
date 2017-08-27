#!/usr/bin/env python

import glob
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from vae_tf.vae import VAE
from vae_tf.clustering import hierarchical_clustering
from vae_tf.utils import random_subset, fc_or_conv_arg
from vae_tf.mnist_helpers import load_mnist

import argparse
parser = argparse.ArgumentParser(description='Run classification')
parser.add_argument('--beta', nargs=1, type=float, default=[None],
                    help='If given, this values will be written to the accuracies.txt file')
parser.add_argument('--arch', nargs='*', type=fc_or_conv_arg, default=[None],
                    help='If given, this list will be written to the accuracies.txt file')
parser.add_argument('--save', type=str, default='./accuracies.txt',
                    help='Where to save accuracies (appending)')
parser.add_argument('--cluster_test', action='store_true',
                    help='Classify by clustering encoded test data (network trained on train data)')
parser.add_argument('--cluster_test_in', action='store_true',
                    help='Classify test data in input space')
parser.add_argument('--cluster_train', action='store_true',
                    help='Classify encoded test data by using centroids from clustering encoded train data')
parser.add_argument('--repeat', type=int, default=1,
                    help='How often to repeat the classification with same datasets (but new sampling in latent space).')
parser.add_argument('--log_folder', type=str, default=None,
                    help='Where the vae model log folders are stored.')
args = parser.parse_args()

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if args.log_folder:
    LOG_FOLDER = 'log_' + args.log_folder
else:
    LOG_FOLDER = 'log'

model_label = '\t'.join([str(args.arch), str(args.beta[0])])

savefile = args.save
filename, ext = os.path.splitext(savefile)
savefile_single_runs = filename + '_detailed' + ext

if not os.path.isfile(savefile):
    header_txt = '''# arch: architecture as used as parameter in VAE class in vae.py
# beta: beta value of beta-VAE
# stat: statistic (mean or std)
# num_runs: how many runs were used to calc mean and std
# cluster_test_latent: classification accuracy by clustering encoded test data (network trained on train data)
# cluster_test_input: classification accuracy by clustering test data in input space (without training or encoding)
# cluster_train_latent: classification accuracy by classifying encoded test data using centroids from clustering encoded train data'''
    with open(savefile, 'w') as log_file:
        column_labels = '\t'.join(['arch', 'beta', 'stat', 'num_runs',
                                   'clust_test_latent',
                                   'clust_test_input',
                                   'clust_train_latent'])
        log_file.write(header_txt + '\n' + column_labels + '\n')

if not os.path.isfile(savefile_single_runs):
    header_txt = '''# arch: architecture as used as parameter in VAE class in vae.py
# beta: beta value of beta-VAE
# cluster_test_latent: classification accuracy by clustering encoded test data (network trained on train data)
# cluster_test_input: classification accuracy by clustering test data in input space (without training or encoding)
# cluster_train_latent: classification accuracy by classifying encoded test data using centroids from clustering encoded train data'''
    with open(savefile_single_runs, 'w') as log_file:
        column_labels = '\t'.join(['arch', 'beta', 'clust_test_latent', 'clust_test_input', 'clust_train_latent'])#classification_latent_with_train_centroids'])
        log_file.write(header_txt + '\n' + column_labels + '\n')

# load most recently trained vae model (assuming it hasen't been first reloaded)
log_folders = [path for path in glob.glob('./' + LOG_FOLDER + '/*') if not 'reloaded' in path]
log_folders.sort(key=os.path.getmtime)
META_GRAPH = log_folders[-1]

# specify log folder manually
#META_GRAPH = './log_CNN/170713_1213_vae_784_conv-20x5x5-2x2-S_conv-50x5x5-2x2-S_conv-70x5x5-2x2-S_conv-100x5x5-2x2-S_10/'

# load mnist datasets
mnist = load_mnist()

# load trained VAE from last checkpoint
last_ckpt = tf.train.latest_checkpoint(META_GRAPH)
assert last_ckpt is not None, "No checkpoint found in {}".format(META_GRAPH)
last_ckpt_path = os.path.abspath(last_ckpt)

#TODO: should this be an option? This results in the classification being the same for
# each repition (since the variance comes from drawing sample in latent space)
# seed = np.random.randint()  # TODO what should be the range of the seed?
seed = None#1234

accuracy_cluster_test_list = []
accuracy_cluster_train_list = []
accuracy_cluster_test_in_list = []
label_list = [mnist.test.labels]
label_names=['true_label']
for n in range(args.repeat):
    tf.reset_default_graph()
    print('\n\nSTARTING CLUSTERING ROUND {}\n'.format(n))
    print('Loading trained vae model from {}'.format(last_ckpt_path))
    vae = VAE(meta_graph=last_ckpt_path)

    ## CLASSIFY BY CLUSTERING ENCODED TEST DATA (network trained on train data)
    if args.cluster_test:
        # encode mnist.test into latent space for clustering
        # TODO any consequences of sampling here?
        test_latent = vae.sesh.run(vae.sampleGaussian(*vae.encode(mnist.test.images), seed=seed))

        # do clustering
        cluster_test = hierarchical_clustering(
            test_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
            plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
            num_clusters=10, max_dist=None, true_labels=mnist.test.labels
        )
        label_list.append(cluster_test.labels)
        label_names.append('hierarchical_clustering')

        # calculate classification accuracy
        accuracy_cluster_test = np.sum(mnist.test.labels == cluster_test.labels) / mnist.test.num_examples
        print('Classification accuracy after HIERARCHICAL CLUSTERING of TEST data in LATENT space is '
              '{}\n'.format(accuracy_cluster_test))
    else:
        accuracy_cluster_test = np.nan
    accuracy_cluster_test_list.append(accuracy_cluster_test)


    if args.cluster_test_in:
        # do clustering in input space
        shape = mnist.test.images.shape
        test_images_flat = mnist.test.images.reshape((shape[0], shape[1] * shape[2] * shape[3]))
        cluster_test_in = hierarchical_clustering(
            test_images_flat, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
            plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
            num_clusters=10, max_dist=None, true_labels=mnist.test.labels
        )
        label_list.append(cluster_test_in.labels)
        label_names.append('hierarchical_clustering_x_input')

        # calculate classification accuracy
        accuracy_cluster_test_in = np.sum(mnist.test.labels == cluster_test_in.labels) / mnist.test.num_examples
        print('Classification accuracy after HIERARCHICAL CLUSTERING of TEST data in INPUT space is '
              '{}\n'.format(accuracy_cluster_test_in))
    else:
        accuracy_cluster_test_in = np.nan
    accuracy_cluster_test_in_list.append(accuracy_cluster_test_in)



    ## CLASSIFY ENCODED TEST DATA BY USING CENTROIDS FROM CLUSTERING ENCODED TRAIN DATA
    if args.cluster_train:
        ## (network trained on train data)
        # encode mnist.train into latent space for clustering
        # TODO any consequences of sampling here?
        train_latent = vae.sesh.run(vae.sampleGaussian(*vae.encode(mnist.train.images), seed=seed))

        # do clustering
        cluster_train = hierarchical_clustering(
            train_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
            plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
            num_clusters=10, max_dist=None, true_labels=mnist.train.labels
        )
        # classify test data by closest cluster centroid
        classify_test_labels = cluster_train.assign_to_nearest_cluster(test_latent)
        label_names.append('classification_from_train_clusters')

        label_list.append(classify_test_labels)

        # calculate classification accuracy
        accuracy_cluster_train = np.sum(mnist.test.labels == classify_test_labels) / mnist.test.num_examples
        print('Classification accuracy after CLASSIFICATION of TEST data in LATENT space using nearest '
              'centroids from hierarchical clustering of train data in latent space is '
              '{}\n'.format(accuracy_cluster_train))
    else:
        accuracy_cluster_train = np.nan
    accuracy_cluster_train_list.append(accuracy_cluster_train)

    # create embedding with cluster labels of subset of subset_size samples
    labels = np.vstack(label_list).T
    subset_size = 1000
    subset_test_images, subset_labels = random_subset(mnist.test.images, subset_size, labels=labels,
                                                      same_num_labels=True)
    vae.create_embedding(subset_test_images, labels=subset_labels, label_names=label_names,
                         sample_latent=True, latent_space=True, input_space=True,
                         image_dims=(28, 28))

    with open(savefile_single_runs, 'a') as log_file:
        txt = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(str(args.arch), str(args.beta[0]),
                                                        accuracy_cluster_test, accuracy_cluster_train,
                                                        accuracy_cluster_test_in)
        print('Saving single run clustering accuracies in {}'.format(savefile_single_runs))
        log_file.write(txt)

accuracy_mean = np.mean(accuracy_cluster_test_list)
accuracy_std = np.std(accuracy_cluster_test_list)
accuracy_cluster_train_mean = np.mean(accuracy_cluster_train_list)
accuracy_cluster_train_std = np.std(accuracy_cluster_train_list)
accuracy_cluster_test_in_mean = np.mean(accuracy_cluster_test_in_list)
accuracy_cluster_test_in_std = np.std(accuracy_cluster_test_in_list)

with open(savefile, 'a') as log_file:
    mean_txt = '{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(str(args.arch), str(args.beta[0]), 'mean', args.repeat,
                                                                 accuracy_mean,
                                                                 accuracy_cluster_train_mean,
                                                                 accuracy_cluster_test_in_mean)
    log_file.write(mean_txt)
    std_txt = '{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(str(args.arch), str(args.beta[0]), 'std', args.repeat,
                                                                 accuracy_std,
                                                                 accuracy_cluster_train_std,
                                                                 accuracy_cluster_test_in_std)
    log_file.write(std_txt)
    print('Saved mean and std of clustering accuracies in {}'.format(savefile))
