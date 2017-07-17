import glob
import os
import sys

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree, KDTree
import numpy as np
import tensorflow as tf

from vae import VAE
from clustering import ClusteringResults, hierarchical_clustering
from utils import random_subset
from main import load_mnist

import argparse
parser = argparse.ArgumentParser(description='Run classification')
parser.add_argument('--beta', nargs=1, type=float,
                    help='If given, this values will be written to the accuracies.txt file')
parser.add_argument('--arch', nargs='*', type=int,
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
parser.add_argument('--log_folder', type=str, default='./log',
                    help='Where the vae model log folders are stored.')
args = parser.parse_args()

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load most recently trained vae model (assuming it hasen't been first reloaded)
log_folders = [path for path in glob.glob('./' + args.log_folder + '/*') if not 'reloaded' in path]
log_folders.sort(key=os.path.getmtime)
META_GRAPH = log_folders[-1]

# specify log folder manually
#META_GRAPH = './log_CNN/170713_1213_vae_784_conv-20x5x5-2x2-S_conv-50x5x5-2x2-S_conv-70x5x5-2x2-S_conv-100x5x5-2x2-S_10/'

# load mnist datasets
mnist = load_mnist()

# load trained VAE from last checkpoint
last_ckpt_path = os.path.abspath(tf.train.latest_checkpoint(META_GRAPH))

for _ in range(args.repeat):
    tf.reset_default_graph()
    print('Loading trained vae model from {}'.format(last_ckpt_path))
    vae = VAE(meta_graph=last_ckpt_path)

    ## CLASSIFY BY CLUSTERING ENCODED TEST DATA (network trained on train data)
    if args.cluster_test:
        # encode mnist.test into latent space for clustering
        # TODO any consequences of sampling here?
        test_latent = vae.sesh.run(vae.sampleGaussian(*vae.encode(mnist.test.images)))

        # do clustering
        cluster_test = hierarchical_clustering(
            test_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
            plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
            num_clusters=10, max_dist=None, true_labels=mnist.test.labels
        )
    else:
        cluster_test = ClusteringResults(np.zeros(mnist.test.num_examples), None)

    # calculate classification accuracy
    accuracy = np.sum(mnist.test.labels == cluster_test.labels) / mnist.test.num_examples
    print('Classification accuracy LATENT after hierachical clustering of test data is {}'.format(accuracy))

    if args.cluster_test_in:
        # do clustering in input space
        shape = mnist.test.images.shape
        test_images = mnist.test.images.reshape((shape[0], shape[1] * shape[2] * shape[3]))
        cluster_test_in = hierarchical_clustering(
            test_images, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
            plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
            num_clusters=10, max_dist=None, true_labels=mnist.test.labels
        )

    else:
        cluster_test_in = ClusteringResults(np.zeros(mnist.test.num_examples), None)

    # calculate classification accuracy
    accuracy_in = np.sum(mnist.test.labels == cluster_test_in.labels) / mnist.test.num_examples
    print('Classification accuracy INPUT after hierachical clustering of test data is {}'.format(accuracy_in))


    ## CLASSIFY ENCODED TEST DATA BY USING CENTROIDS FROM CLUSTERING ENCODED TRAIN DATA
    if args.cluster_train:
        ## (network trained on train data)
        # encode mnist.train into latent space for clustering
        # TODO any consequences of sampling here?
        train_latent = vae.sesh.run(vae.sampleGaussian(*vae.encode(mnist.train.images)))

        # do clustering
        cluster_train= hierarchical_clustering(
            train_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
            plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
            num_clusters=10, max_dist=None, true_labels=mnist.train.labels
        )

        # classify test data by closest cluster centroid
        kdtree = cKDTree(cluster_train.centroids)
        dist, classify_test_labels = kdtree.query(test_latent, n_jobs=8)
        #kdtree = KDTree(cluster_train.centroids)
        #dist, classify_test_labels = kdtree.query(test_latent)

    else:
        classify_test_labels = np.zeros(mnist.test.num_examples)

    # calculate classification accuracy
    accuracy2 = np.sum(mnist.test.labels == classify_test_labels) / mnist.test.num_examples
    print('Classification accuracy LATENT after classification of test data with train data clusters is {}'.format(accuracy2))

    # create embedding with cluster labels
    # subset of size samples
    labels = np.vstack([mnist.test.labels, cluster_test.labels, classify_test_labels, cluster_test_in.labels]).T
    size = 1000
    subset_test_images, subset_labels = random_subset(mnist.test.images, size, labels=labels, same_num_labels=True)
    vae.create_embedding(subset_test_images, labels=subset_labels,
                         label_names=['true_label', 'hierarchical_clustering', 'classification_from_train_clusters',
                                      'hierarchical_clustering_x_input'],
                         sample_latent=True, latent_space=True, input_space=True,
                         image_dims=(28, 28))

    label = []
    if args.arch:
        label.append(str(args.arch))

    if args.beta:
        label.append(str(args.beta[0]))

    label = '\t'.join(label)

    with open(args.save, 'a') as log_file:
        txt = '{}\t{}\t{}\t{}\n'.format(label, accuracy, accuracy2, accuracy_in)
        log_file.write(txt)
