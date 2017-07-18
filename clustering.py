import glob
import os

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree, KDTree
import numpy as np

from vae import VAE
from plot import fancy_dendrogram
from main import load_mnist


class ClusteringResults(object):

    def __init__(self, labels, centroids):
        self.labels = labels
        self.centroids = centroids

    def assign_to_nearest_cluster(self, data, n_jobs=-1):
        # classify test data by closest cluster centroid
        kdtree = cKDTree(self.centroids)
        #kdtree = KDTree(self.centroids)
        distances, assignments = kdtree.query(data, n_jobs=n_jobs)
        return assignments


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

    #print('cluster indices', np.unique(cluster_indices))
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

        # assign labels in order of their "confidence" (highest_label_fraction)
        sort_idx = np.argsort(highest_label_fractions)[::-1]
        sorted_highest_label_fractions = list(np.array(highest_label_fractions)[sort_idx])
        sorted_unique_cluster_indices = list(unique_cluster_indices[sort_idx])
        cluster_labels = - np.ones(cluster_indices.shape)
        used_labels = {}
        # sort label_options in descending order
        label_options_per_cluster = [list(fractions.argsort()[::-1]) for fractions in label_fractions]
        # outer loop through cluster indices
        num_iter_outer = len(sorted_unique_cluster_indices)
        # offset in label_options array for each cluster (see below)
        idx_offsets_in_label_options = [0] * num_iter_outer
        for n, idx in enumerate(sorted_unique_cluster_indices):
            idx -= 1  # cluster indices start at 1
            fractions = label_fractions[idx]
            highest_fraction = highest_label_fractions[idx]
            #label_options = fractions.argsort()[::-1]
            label_options = label_options_per_cluster[idx]
            # inner loop through label options
            offset = idx_offsets_in_label_options[n]
            num_iter_inner = len(label_options) - offset
            for m, label in enumerate(label_options[offset:]):
                fraction = fractions[label]
                if label not in used_labels.keys():
                    cluster_labels[cluster_masks[idx]] = label
                    used_labels[label] = (idx, fraction)
                    print('{}.{} Cluster {} gets label {} with fraction '
                          '{:.3f} (max {:.3f})'.format(n, m, idx, label, fraction,
                                                       fractions.max()))
                    break
                else:
                    other_idx, other_fraction = used_labels[label]
                    print('\n{}.{} WARNING: label {} already used\n'
                          '\t\t\tthis\tother\n'
                          '\tidx\t\t{}\t{}\n'
                          '\tfraction\t{:.3f}\t{:.3f}\n'.format(n, m, label, idx, other_idx,
                                                                fraction, other_fraction))
                    assert other_fraction > fraction
                    if n == num_iter_outer - 1:
                        # last iteration of outer loop -> just loop through inner loop
                        continue
                    assert m < num_iter_inner - 1, \
                            'Finished inner loop without cluster assignment'
                    assert m<1000 and n<1000, 'Stuck in the loop? Bug?'
                    # make sure that the next assigned label is assigned to most confident
                    # cluster (highest_label_fraction)
                    next_idx = sorted_unique_cluster_indices[n+1]
                    next_idx -= 1  # cluster indices start at 1
                    next_idx_highest_fraction = highest_label_fractions[next_idx]
                    this_idx_next_label = label_options[offset+m+1]
                    this_idx_next_label_fraction = fractions[this_idx_next_label]
                    if this_idx_next_label_fraction >= next_idx_highest_fraction:
                        # continue with inner loop
                        # (equivalent to inserting idx in outer loop at position 0)
                        print('\tThis idx next label fraction >= next idx highest fraction, continue with next label.\n'.format(
                            this_idx_next_label_fraction, next_idx_highest_fraction
                        ))
                        continue
                    else:
                        # sorted_... variables are sorted in descending order ([::-1] -> ascending)
                        # find position in ascending highest_label_fractions
                        insert_position = np.searchsorted(sorted_highest_label_fractions[::-1],
                                                          this_idx_next_label_fraction)
                        # get position in descending highest_label_fractions
                        insert_position = len(sorted_highest_label_fractions) - insert_position
                        # insert cluster idx (note: starts at 1) into outer loop list
                        sorted_unique_cluster_indices.insert(insert_position, idx+1)
                        # insert fraction into highest label fraction list (for the position
                        # search to still work next time we enter this code block)
                        sorted_highest_label_fractions.insert(insert_position, this_idx_next_label_fraction)
                        # keep track that our loop list just got bigger
                        num_iter_outer += 1
                        # next time skip labels up to the current label from label options
                        offset += m + 1
                        # insert the offset into the offset list
                        idx_offsets_in_label_options.insert(insert_position, offset)
                        # 
                        # TODO what happens when we enter for same idx second time here?
                        # delete labels up to the current label from label options
                        #del label_options_per_cluster[idx][:m]
                        print('\tReinserting this idx {} with next label fraction {:.3f} in outer loop at {}.\n'.format(
                            idx, this_idx_next_label_fraction, insert_position
                        ))
                        # and break out of inner loop
                        break
        assert not any(cluster_labels == -1), 'Not all cluster_labels where set!'
    else:  # no true_labels given
        cluster_labels = cluster_indices - 1  # let indexing start at 0
	   
    # calculate centroids (sorted by cluster_labels)
    centroids = []
    for i in range(num_clusters):
        centroids.append(X[cluster_labels == i].mean(0))
    centroids = np.vstack(centroids)

    return ClusteringResults(cluster_labels, centroids)
