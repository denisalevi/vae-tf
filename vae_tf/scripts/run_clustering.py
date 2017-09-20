#!/usr/bin/env python

import glob
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from vae_tf.vae import VAE
from vae_tf.clustering import hierarchical_clustering, kmeans_clustering
from vae_tf.utils import random_subset, fc_or_conv_arg
from vae_tf.mnist_helpers import load_mnist
from vae_tf.plot import morph, explore_latent_space_dimensions

import argparse
parser = argparse.ArgumentParser(description='Run classification')
parser.add_argument('--beta', nargs=1, type=float, default=[None],
                    help='If given, this values will be written to the accuracies.txt file')
parser.add_argument('--arch', nargs='*', type=str, default=None,
                    help='If given, this list will be written to the accuracies.txt file')
parser.add_argument('--save', type=str, default='./accuracies.txt',
                    help='Where to save accuracies (appending)')
parser.add_argument('--cluster', nargs='+', default=['test_latent'],
                    choices=['test_latent', 'test_input', 'train_latent', 'all'],
                    help='Choose what to cluster.')
parser.add_argument('--method', type=str, default='hierarchical', choices=['kmeans', 'hierarchical'],
                    help='Cluster method to use')
parser.add_argument('--repeat', type=int, default=1,
                    help='How often to repeat the classification with same datasets (but new sampling in latent space).')
parser.add_argument('--log_folder', type=str, default=None,
                    help='Where the vae model log folders are stored.')
parser.add_argument('--meta_graph', type=str, default=None,
                    help='Specific model folder name')
parser.add_argument('--sample_latent', action='store_true',
                    help='Sample data in latent space before clustering (otherwise just use mean for clustering)')
parser.add_argument('--comment', default='-', type=str,
                    help='Comment to put in comment index column in accuracy file. To label same parameter runs differently.')
parser.add_argument('--assign_with_count', action='store_true',
                    help='Use count of label occurences instead of fraction for assignment order.')
parser.add_argument('--use_gpu', action='store_true',
                    help='Only use CPU for comutations.')
parser.add_argument('--num_clusters', type=int, default=10,
                    help='Changes where the hyrarchical clustering is cut to create the correct number of clusters.')
parser.add_argument('--assign_by_prob', action='store_true',
                    help='Assign cluster labels from data point that has the highest probability to belong to that cluster.')
parser.add_argument('--allow_same_labels', action='store_false',  # will be passed as distinct_labels argument
                    help='Allow multiple clusters to get the same cluster label')
parser.add_argument('--save_pngs', nargs='?', default=None, type=str, const=True, 
                    help='Save figures as pngs. Optionally pass target folder as argument.')
args = parser.parse_args()

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES="] = ""

if args.log_folder:
    LOG_FOLDER = 'log_' + args.log_folder
else:
    LOG_FOLDER = 'log'

if 'all' in args.cluster:
    args.cluster = ['test_latent', 'test_input', 'train_latent']

if not args.sample_latent and args.repeat > 1:
    print("WARNING: repeating clustering without sampling in latent space. Waste of time, the results should be deterministic!")

savefile = args.save
filename, ext = os.path.splitext(savefile)
savefile_single_runs = filename + '_detailed' + ext

if not os.path.isfile(savefile):
    header_txt = '''# arch: architecture as used as parameter in VAE class in vae.py
# beta: beta value of beta-VAE
# comment: extra index column for comment
# stat: statistic (mean or std)
# num_runs: how many runs were used to calc mean and std
# cluster_test_latent: classification accuracy by clustering encoded test data (network trained on train data)
# cluster_test_input: classification accuracy by clustering test data in input space (without training or encoding)
# cluster_train_latent: classification accuracy by classifying encoded test data using centroids from clustering encoded train data'''
    with open(savefile, 'w') as log_file:
        column_labels = '\t'.join(['arch', 'beta', 'comment', 'stat', 'num_runs',
                                   'clust_test_latent',
                                   'clust_train_latent',
                                   'clust_test_input'])
        log_file.write(header_txt + '\n' + column_labels + '\n')

if not os.path.isfile(savefile_single_runs):
    header_txt = '''# arch: architecture as used as parameter in VAE class in vae.py
# beta: beta value of beta-VAE
# comment: extra index column for comment
# cluster_test_latent: classification accuracy by clustering encoded test data (network trained on train data)
# cluster_test_input: classification accuracy by clustering test data in input space (without training or encoding)
# cluster_train_latent: classification accuracy by classifying encoded test data using centroids from clustering encoded train data'''
    with open(savefile_single_runs, 'w') as log_file:
        column_labels = '\t'.join(['arch', 'beta', 'comment',
                                   'clust_test_latent',
                                   'clust_train_latent',
                                   'clust_test_input'])
        log_file.write(header_txt + '\n' + column_labels + '\n')

if args.meta_graph is not None:
    META_GRAPH = args.meta_graph
else:
    # load most recently trained vae model (assuming it hasen't been first reloaded)
    log_folders = [path for path in glob.glob('./' + LOG_FOLDER + '/*') if not 'reloaded' in path]
    log_folders.sort(key=os.path.getmtime)
    META_GRAPH = log_folders[-1]

# specify log folder manually
#META_GRAPH = './log_CNN/170713_1213_vae_784_conv-20x5x5-2x2-S_conv-50x5x5-2x2-S_conv-70x5x5-2x2-S_conv-100x5x5-2x2-S_10'

# get arch and beta from META_GRAPH string
# add trialing slash if not there (turns 'path/to/folder' and 'path/to/folder/' into 'path/to/folder/')
model_name = os.path.join(META_GRAPH, '')
# remove trailing slash (returns 'path/to/folder')
model_name = os.path.dirname(model_name)
# get the folder name (returns 'folder')
model_name = os.path.basename(model_name)
# get model architecture
prefix, model_name = model_name.split("_vae_")
arch_from_name, beta_from_name, img_dim_from_name = VAE.get_architecture_from_model_name(model_name)

# TODO do we want img dims as cluster column?

if args.beta[0] is None:
    beta = beta_from_name
else:
    beta = args.beta[0]

if args.arch is None:
    size = []
    filt = []
    stride = []
#    for layer in arch_from_name[1:-1]:
#        size.append(layer[0])
#        filt.append(layer[1][0])
#        stride.append(layer[2][0])
#    arch = str(size) + ' F{}'.format(filt[0]) + ' S{}'.format(np.unique(stride)) + ' {}'.format(arch_from_name[-1])
    arch = None
else:
    arch = args.arch

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

all_clust_results = {}
last_clust_accuracies = {}
accuracy_clust_test_latent_list = []
accuracy_clust_train_latent_list = []
accuracy_clust_test_input_list = []
label_list = [mnist.test.labels]
label_names=['true_label']
for n in range(args.repeat):
    tf.reset_default_graph()
    print('\n\nSTARTING CLUSTERING ROUND {}\n'.format(n))
    print('Loading trained vae model from {}'.format(last_ckpt_path))
    vae = VAE(meta_graph=last_ckpt_path)

    if 'test_latent' in args.cluster or 'train_latent' in args.cluster:
        # encode mnist.test into latent space for clustering
        mu, log_sigma = vae.encode(mnist.test.images)
        if args.sample_latent:
            test_latent = vae.sesh.run(vae.sampleGaussian(mu, log_sigma, seed=seed))
        else:
            # cluster using only the mean
            test_latent = mu

    ## CLASSIFY BY CLUSTERING ENCODED TEST DATA (network trained on train data)
    if 'test_latent' in args.cluster:
        # do clustering of test data in latent space
        if args.method == 'hierarchical':
            clust_test_latent = hierarchical_clustering(
                test_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
                plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
                num_clusters=args.num_clusters, max_dist=None, true_labels=mnist.test.labels,
                assign_clusters_with_label_count=args.assign_with_count,
                assign_labels_by_probability=args.assign_by_prob,
                distinct_labels=args.allow_same_labels
            )
        else:  # args.method == 'kmeans'
            clust_test_latent = kmeans_clustering(
                test_latent, args.num_clusters, true_labels=mnist.test.labels,
                assign_clusters_with_label_count=args.assign_with_count,
                assign_labels_by_probability=args.assign_by_prob,
                distinct_labels=args.allow_same_labels
            )
        label_list.append(clust_test_latent.data_labels)
        label_names.append('{}_clustering'.format(args.method))

        # calculate classification accuracy
        accuracy_clust_test_latent = np.sum(mnist.test.labels == clust_test_latent.data_labels) / mnist.test.num_examples
        print('Classification accuracy after {} CLUSTERING of TEST data in LATENT space is '
              '{}\n'.format(args.method.upper(), accuracy_clust_test_latent))

        all_clust_results['test_latent'] = clust_test_latent
        last_clust_accuracies['test_latent'] = accuracy_clust_test_latent
    else:
        accuracy_clust_test_latent = np.nan
    accuracy_clust_test_latent_list.append(accuracy_clust_test_latent)


    if 'test_input' in args.cluster:
        # do clustering in input space
        shape = mnist.test.images.shape
        test_images_flat = mnist.test.images.reshape((shape[0], shape[1] * shape[2] * shape[3]))
        if args.method == 'hierarchical':
            clust_test_input = hierarchical_clustering(
                test_images_flat, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
                plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
                num_clusters=args.num_clusters, max_dist=None, true_labels=mnist.test.labels,
                assign_clusters_with_label_count=args.assign_with_count,
                assign_labels_by_probability=args.assign_by_prob,
                distinct_labels=args.allow_same_labels
            )
        else:  # args.method == 'kmeans'
            clust_test_input = kmeans_clustering(
                test_images_flat, args.num_clusters, true_labels=mnist.test.labels,
                assign_clusters_with_label_count=args.assign_with_count,
                assign_labels_by_probability=args.assign_by_prob,
                distinct_labels=args.allow_same_labels
            )
        label_list.append(clust_test_input.data_labels)
        label_names.append('{}_clustering_x_input'.format(args.method))

        # calculate classification accuracy
        accuracy_clust_test_input = np.sum(mnist.test.labels == clust_test_input.data_labels) / mnist.test.num_examples
        print('Classification accuracy after {} CLUSTERING of TEST data in INPUT space is '
              '{}\n'.format(args.method.upper(), accuracy_clust_test_input))
    else:
        accuracy_clust_test_input = np.nan
    accuracy_clust_test_input_list.append(accuracy_clust_test_input)



    ## CLASSIFY ENCODED TEST DATA BY USING CENTROIDS FROM CLUSTERING ENCODED TRAIN DATA
    if 'train_latent' in args.cluster:
        ## (network trained on train data)
        # encode mnist.train into latent space for clustering
        mu, log_sigma = vae.encode(mnist.train.images)
        if args.sample_latent:
            train_latent = vae.sesh.run(vae.sampleGaussian(mu, log_sigma, seed=seed))
        else:
            # cluster using only the mean
            train_latent = mu

        # do clustering
        if args.method == 'hierarchical':
            clust_train_latent = hierarchical_clustering(
                train_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
                plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
                num_clusters=args.num_clusters, max_dist=None, true_labels=mnist.train.labels,
                assign_clusters_with_label_count=args.assign_with_count,
                assign_labels_by_probability=args.assign_by_prob,
                distinct_labels=args.allow_same_labels
            )
        else:  # args.method == 'kmeans'
            clust_train_latent = kmeans_clustering(
                train_latent, args.num_clusters, true_labels=mnist.train.labels,
                assign_clusters_with_label_count=args.assign_with_count,
                assign_labels_by_probability=args.assign_by_prob,
                distinct_labels=args.allow_same_labels
            )
        # classify test data by closest cluster centroid
        classify_test_labels = clust_train_latent.classify_by_nearest_cluster(test_latent)
        label_names.append('classification_from_train_clusters')

        label_list.append(classify_test_labels)

        # calculate classification accuracy
        accuracy_clust_train_latent = np.sum(mnist.test.labels == classify_test_labels) / mnist.test.num_examples
        print('Classification accuracy after CLASSIFICATION of TEST data in LATENT space using nearest '
              'centroids from {} clustering of train data in latent space is '
              '{}\n'.format(args.method, accuracy_clust_train_latent))

        all_clust_results['train_latent'] = clust_train_latent
        last_clust_accuracies['train_latent'] = accuracy_clust_train_latent
    else:
        accuracy_clust_train_latent = np.nan
    accuracy_clust_train_latent_list.append(accuracy_clust_train_latent)

    # create embedding with cluster labels of subset of subset_size samples
    labels = np.vstack(label_list).T
    subset_size = 1000
    subset_test_images, subset_labels = random_subset(mnist.test.images, subset_size, labels=labels,
                                                      same_num_labels=True)
    vae.create_embedding(subset_test_images, labels=subset_labels, label_names=label_names,
                         sample_latent=False, latent_space=True, input_space=True,
                         create_sprite=True)

    with open(savefile_single_runs, 'a') as log_file:
        txt = '{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(str(arch), str(beta), args.comment,
                                                            accuracy_clust_test_latent,
                                                            accuracy_clust_train_latent,
                                                            accuracy_clust_test_input)
        print('Saving single run clustering accuracies in {}'.format(savefile_single_runs))
        log_file.write(txt)


## Latent space visualisation around cluster centroids
with tf.variable_scope('latent_space'):
    # morph btwn cluster centroids for each cluster type (only last run if args.repeat)
    for cluster_name in all_clust_results.keys():
        cluster_result = all_clust_results[cluster_name]
        accuracy = last_clust_accuracies[cluster_name]
        print('Morphing latent space cluster centroid from clustering {}'.format(cluster_name))
        morph(vae, cluster_result.cluster_centroids,
              name='morph_centroids_{}_{}'.format(cluster_name, accuracy),
              tf_summary=True, save_png=args.save_pngs)

    # explore single latent dimensions for a cluster centroid from the run with highest accuracy
    # choose clustering centroids from cluster run with highest accuracy
    cluster_name = max(last_clust_accuracies, key=last_clust_accuracies.get)
    cluster_result = all_clust_results[cluster_name]
    # choose random cluster centroid
    centroid_idx = np.random.randint(0, args.num_clusters)
    amplitude = 5
    print('Morphing around single latent dimensions of cluster centroid {} from clustering {}'
          .format(centroid_idx, cluster_name))
    explore_latent_space_dimensions(vae, amplitude, n=9,
                                    origin=cluster_result.cluster_centroids[centroid_idx],
                                    name='explore_latent_dims_{}_centroid-{}_amp-{}'.format(
                                        cluster_name, centroid_idx, amplitude),
                                    tf_summary=vae.validation_writer_dir,
                                    save_png=args.save_pngs)


accuracy_clust_test_latent_mean = np.mean(accuracy_clust_test_latent_list)
accuracy_clust_test_latent_std = np.std(accuracy_clust_test_latent_list)
accuracy_clust_train_latent_mean = np.mean(accuracy_clust_train_latent_list)
accuracy_clust_train_latent_std = np.std(accuracy_clust_train_latent_list)
accuracy_clust_test_input_mean = np.mean(accuracy_clust_test_input_list)
accuracy_clust_test_input_std = np.std(accuracy_clust_test_input_list)

with open(savefile, 'a') as log_file:
    mean_txt = '{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
        str(arch), str(beta), args.comment, 'mean', args.repeat,
        accuracy_clust_test_latent_mean,
        accuracy_clust_train_latent_mean,
        accuracy_clust_test_input_mean
    )
    log_file.write(mean_txt)
    std_txt = '{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
        str(arch), str(beta), args.comment, 'std', args.repeat,
        accuracy_clust_test_latent_std,
        accuracy_clust_train_latent_std,
        accuracy_clust_test_input_std
    )
    log_file.write(std_txt)
    print('Saved mean and std of clustering accuracies in {}'.format(savefile))
