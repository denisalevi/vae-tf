import glob
import os

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np

from vae import VAE
from plot import fancy_dendrogram
from main import load_mnist


class ClusteringResults(object):
    def __init__(self, labels, centroids):
        self.labels = labels
        self.centroids = centroids


def hierarchical_clustering(X, linkage_method='ward', distance_metric='euclidean',
                            check_cophonetic=False, plot_dir=None,
                            truncate_dendrogram=None, num_clusters=None,
                            max_dist=None, true_labels=None):
    """hierarchical_clustering

    :param X: dataset in clustering space, shape (num_samples, num_dimensions)
    :param linkage_method: scipy.cluster.hierarchy.linkage(..., method=linkage_method)
    :param distance_metric: scipy.cluster.hierarchy.linkage(..., metric=distance_metric)
    :param check_cophonetic: if True, prints Cophonetic Correlation Coefficient (memory intensive)
    :param plot_dir: where to save dendrogram plot, default None (no plot created)
    :param truncate_dendrogram: int or None, how many merged clusters to show (None shows all)
    :param num_clusters: determines where to cut the dendrogram for cluster assignment
    :param max_dist: determines where to cut dendrogram for cluster assignment
    :param true_labels: if given, cluster labels are chosen from highest occuring true_labels in cluster
    """
    if num_clusters is not None and max_dist is not None:
        raise ValueError("You can't specifyg num_clusters and max_dist.")
    if num_clusters is None and max_dist is None:
        print('WARNING: Neither num_cluster nor max_dist are given. '
              'Just printing dendrogram without assigning data to clusters.')

    # create plot dir
    try:
        os.makedirs(plot_dir)
    except FileExistsError:
        pass

    # generate the linkage matrix
    Z = linkage(X, method=linkage_method, metric=distance_metric)
    
    if check_cophonetic:
        # check Cophonetic Correlation Coefficient, showing how faithfully pairwise
        # distances are preserved by the dendrogramm (0 bad, 1 perfect)
        c, coph_dists = cophenet(Z, pdist(X))
        print('Cophonetic Correlation Coefficient: {:.4f}'.format(c))
        # delete c to free memory
        del c, coph_dists
    
    ## PLOTS
    # FULL DENDROGRAM
    #plt.figure(figsize=(25, 10))
    #plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('sample index')
    #plt.ylabel('distance')
    #dendrogram(
    #    Z,
    #    leaf_rotation=90.,  # rotates the x axis labels
    #    leaf_font_size=8.,  # font size for the x axis labels
    #    labels=true_labels
    #)
    #plt.savefig(os.path.join(PLOT_DIR, 'dendrogram_full.png'))
    #if SHOW_PLOTS:
    #    plt.show()
    #plt.close()
    
    if plot_dir is not None:
        # TRUNCATED DENDROGRAM
        if truncate_dendrogram is not None:
            # show only the last p merged clusters
            truncate_mode = 'lastp'
        else:
            truncate_mode = None  # default, ignores p
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        fancy_dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=truncate_dendrogram,  # show only the last p merged clusters
            show_leaf_counts=False,  # numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        	annotate_above=10,  # useful in small plots so annotations don't overlap
            max_d=max_dist,
            #labels=true_labels
        )
        plt.savefig(os.path.join(plot_dir, 'dendrogram_truncated.png'))
        plt.close()
    
    
    ## GET CLUSTERS
    if num_clusters is not None:
        cluster_indices = fcluster(Z, num_clusters, criterion='maxclust')

    if max_dist is not None:
        cluster_indices = fcluster(Z, max_dist, criterion='distance')
        num_clusters = int(np.max(cluster_indices))

    print('cluster indices', np.unique(cluster_indices))
    print('cluster idx count', np.bincount(cluster_indices))
    
    if true_labels is not None:

        cluster_masks = []
        label_fractions = []
        highest_label_fractions = []
        unique_cluster_indices = np.arange(1, num_clusters + 1)  # starts with 1
        for idx in unique_cluster_indices:
            mask = (cluster_indices == idx)
            cluster_masks.append(mask)
            # get true labels for this cluster
            true_cluster_labels = true_labels[mask]
            # get the ocurrences of true labels in the cluster
            counts = np.bincount(true_cluster_labels)
            # compute their fraction to the total number of labels
            fractions = counts / counts.sum()
            label_fractions.append(fractions)
            highest_label_fractions.append(fractions.max())

        # assign labels in order of their "condidence" (highest_label_fraction)
        sort_idx = np.argsort(highest_label_fractions)[::-1]
        cluster_labels = - np.ones(cluster_indices.shape)
        used_labels = {}
        for idx in unique_cluster_indices[sort_idx]:
            fractions = label_fractions[idx-1]
            label_options = fractions.argsort()[::-1]
            for label in label_options:
                fraction = fractions[label]
                if label not in used_labels.keys():
                    cluster_labels[cluster_masks[idx-1]] = label
                    used_labels[label] = (idx, fraction)
                    print('Cluster {} gets label {} with fraction '
                          '{:.3f} (max {:.3f})'.format(idx, label, fraction,
                                                       fractions.max()))
                    break
                else:
                    other_idx, other_fraction = used_labels[label]
                    #other_fractions = label_fractions[other_idx-1]
                    print('\nWARNING: label {} already used\n'
                          '\t\t\tthis\tother\n'
                          '\tidx\t\t{}\t{}\n'
                          '\tfraction\t{:.3f}\t{:.3f}\n'.format(label, idx,other_idx,
                                                              fraction, other_fraction))
        assert not any(cluster_labels == -1), 'Not all cluster_labels where set!'
    else:  # no true_labels given
        cluster_labels = cluster_indices - 1  # let indexing start at 0
	   
    # calculate centroids (sorted by cluster_labels)
    centroids = []
    for i in range(num_clusters):
        centroids.append(X[cluster_labels == i].mean(0))
    centroids = np.vstack(centroids)

    return ClusteringResults(cluster_labels, centroids)
