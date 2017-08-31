import glob
import os

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree, KDTree
from sklearn.cluster import KMeans
import numpy as np

from vae_tf.vae import VAE
from vae_tf.plot import fancy_dendrogram
from vae_tf.mnist_helpers import load_mnist


class ClusteringResults(object):
    '''
    Class to store and work with clustering results.

    Parameters
    ----------
    data_indices : ndarray
        1D array of `int` corresponding to cluster numbers (in range(1,
        num_clusters + 1)) for clustered data (size `data_size`).
    data_coords : ndarray, optional
        `data_size`x`m` array of data coordinates, where `data_size` is the
        number data points and `m` is the cluster space dimensionality. Used to
        compute `cluster_centroids` if not given.
    cluster_centroids : ndarray, optional
        `num_clusters`x`m` array of cluster centroids, where `num_clusters` is
        the number of clusters and `m` the cluster space dimensionality.
    cluster_labels : ndarray, optional
        1D array of cluster labels with size equal to `num_clusters`.
    num_clusters : int, optional
        The number of clusters. If None (default), the maximum in
        `data_indices` is taken.

    Examples
    --------
    >>> import numpy as np
    >>> data_cluster_indices = np.array([3, 3, 2, 1, 1, 1])

    >>> res = ClusteringResults(data_cluster_indices)
    >>> all(res.data_indices == data_cluster_indices)
    True
    >>> res.data_size
    6
    >>> res.num_clusters
    3
    '''

    def __init__(self, data_indices, data_coords=None, cluster_centroids=None,
                 cluster_labels=None, num_clusters=None):
        self.data_indices = data_indices
        #: The number of data points used for clustering.
        self.data_size = len(data_indices)

        if num_clusters is None:
            self.num_clusters = data_indices.max()  #: The number of clusters.
        else:
            assert isinstance(num_clusters, int),\
                    'num_clusters needs to be `int`, got {}'.format(type(num_clusters))
            self.num_clusters = num_clusters  #: The number of clusters.

        self.cluster_centroids = cluster_centroids
        if self.cluster_centroids is None and data_coords is not None:
            self.compute_cluster_centroids(data_coords)
        elif data_coords is not None:  # cluster_centroids is not None
            print('WARNING: cluster_centroids and data_coords given. Ignoring data_coords.')
        elif self.cluster_centroids is not None:  # data_coords is None
            assert len(self.cluster_centroids) == self.num_clusters
        else:  # cluster_centroids is None and data_coords is None
            assert self.cluster_centroids is None

        if cluster_labels is not None:
            assert len(self.cluster_labels) == self.num_clusters
        self.cluster_labels = cluster_labels

    @property
    def data_labels(self):
        '''ndarray: 1D array of size `data_size` with labels corresponding to
        the cluster label of the cluster the data points belong to.
        '''
        if self.cluster_labels is None:
            print('WARNING: `data_labels` are not assigned, using `data_indices '
                  '- 1` (cluster numbering starting at 0) ' 'instead to compute '
                  '`data_labels`.')
            return self.data_indices - 1
        else:
            return self.cluster_labels[self.data_indices - 1]

    def compute_cluster_centroids(self, data_coords):
        '''
        Compute the cluster centroids from given data coordinates.

        Computes the mean coordinate per cluster number in `data_indices` and
        assignes these centroid coordinates to `cluster_centroids`.

        Parameters
        ----------
        data_coords : ndarray
            `data_size`x`m` array of data coordinates, where `data_size` is the
            number of clustered data points and `m` is the cluster space dimensionality.
        '''
        if self.cluster_centroids is not None:
            print("WARNING: `cluster_centroids` already computed. Overwriting with new results.")
        # calculate centroids (sorted by cluster_number)
        self.cluster_centroids = []
        for cluster_number in range(1, self.num_clusters + 1):  # cluster_number start at 1
            self.cluster_centroids.append(data_coords[self.data_indices == cluster_number].mean(0))
        self.cluster_centroids = np.vstack(self.cluster_centroids)
        print('vstack centroid result shape', self.cluster_centroids.shape)

    def assign_cluster_labels_by_closest_to_centroid(self, data, true_labels, verbose=True):
        '''
        Assign to clusters the label of the data point closest to their centroid.

        This data point has the "highest probability" of belonging to that
        cluster.  The method is following the evaluation protocol described in
        "Makhzani et al. (2015): Adversarial Autoencoder"
        (https://arxiv.org/abs/1511.05644)

        Parameters
        ----------
        data : ndarray
            `data_size`x`m` array where `data_size` is the number of data
            points and `m` is the dimension of the cluster space.
        true_labels : ndarray
            1D array of true labels from which the label assignment for the
            cluster is chosen. Has to be of size `data_size`.
        verbose : bool, optional
            If True (default), print info about cluster assignments.
        '''
        assert self.cluster_centroids is not None,\
                "Cluster centroids are not computed. Can't compute distances."
        assert data.ndim == 2, "`data` needs to be 2 dimensional, is {}".format(data.ndim)
        assert data.shape[0] == self.data_size,\
                "`data` has wrong length, has to be {}, got {}".format(self.data_size,
                                                                       data.shape[0])
        if self.cluster_labels is not None:
            print("WARNING: Overwriting already existing cluster labels!")
        self.cluster_labels = - np.ones(self.num_clusters, dtype=int)
        distances = cdist(self.cluster_centroids, data)  # shape = (num_clusters, num_data)
        for cluster_idx in range(self.num_clusters):
            dist_in_cluster = distances[cluster_idx]
            assert len(dist_in_cluster) == self.data_size
            idx_clostest_to_centroid = dist_in_cluster.argmin()
            new_label = true_labels[idx_clostest_to_centroid]
            self.cluster_labels[cluster_idx] = new_label
            if verbose:
                cluster_number = cluster_idx + 1  # cluster numbers start at 1
                true_labels_this_cluster = true_labels[cluster_number == self.data_indices]
                counts = np.bincount(true_labels_this_cluster, minlength=self.num_clusters)
                fractions = counts / counts.sum()
                print('Cluster {} gets label {} with fraction {:.3f} (max {:.3f} for label {})'
                      .format(cluster_idx, new_label, fractions[new_label], fractions.max(),
                              fractions.argmax()))

        print('Assigned labels: {}'.format(np.unique(self.cluster_labels)))

        assert not any(self.cluster_labels == -1),\
                'Not all cluster labels were set! Assigned labels = {}'.format(
                    np.unique(self.cluster_labels))

    def classify_by_nearest_cluster(self, data, n_jobs=-1):
        '''
        Assign each element from `data` the label of its closest cluster.

        Parameters
        ----------
        data : ndarray
            `n`x`m` array of `n` `m`-dimensional data points, where `m` is the
            dimension of the space in which the clustering is performed.
        n_jobs : int, optional
            The number of processes to use to compute the distances between
            data points and centroids. Is passed to
            `scipy.spatial.cKDTree.query()`.

        Returns
        -------
        ndarray
            1D array of same shape as `data` with assigned cluster labels.
        '''
        # TODO add option to use sklearn's kmeans.predict() method here
        assert self.cluster_centroids is not None,\
                "Cluster centroids are not computed. Can't compute distances."
        # classify test data by closest cluster centroid
        kdtree = cKDTree(self.cluster_centroids)
        #kdtree = KDTree(self.cluster_centroids)
        distances, assignments = kdtree.query(data, n_jobs=n_jobs)
        # TODO needs testing
        if self.cluster_labels is None:
            # assignments starts at 0
            return assignments
        else:
            return self.cluster_labels[assignments]

    def assign_cluster_labels_by_fraction(self, true_labels, distinct_labels=True,
                                          use_label_count=False,
                                          return_data_labels=False):
        '''
        Assign to each cluster the label with highest fraction of occurence.
    
        To each cluster assign the label from `true_labels` that is occuring most
        often (`cluster_labels`). If `distinct_labels` is True, different
        labels are assigned to different clusters. To achieve this, assign
        labels to clusters in descending order of the fraction of occurences of
        the most often occuring label to all labels in that cluster.  If a
        label is already assigned to another cluster, find the fraction of the
        next most often occuring label, postpone the label assignment until all
        clusters with higher fractions of most often occuring labels are
        assigned and then try to assign the label with that next most often
        occuring label.  If that label is also already assigned to another
        cluster, repeat the procedure. This way the total number of correct
        labels in the clustered data is maximized if the cluster sizes are
        equal. For differently sized clusters different labeling might increase
        the overall number of correct labels. If `distinct_labels` is False,
        assign to each cluster the label that is occuring most often.
    
        Parameters
        ----------
        cluster_numbers : ndarray
            1D array of `int` of length equal to the number of data points, giving
            the cluster assingment of data to cluster numbers. Cluster numbers have
            to in the range from 1 to the number of clusters (range(1, num_clusters
            + 1)).
        true_labels : ndarray
            1D array of length equal to the number of data points, giving the true
            labels of the data. True labels have to be in the range from 0 to the
            number of cluster -1 (range(num_clusters)).
        distinct_labels : bool, optional
            If True (default), give every cluster a different label. If False,
            multiple clusters can have the same label and not all labels have to
            get assigned to a cluster.
        use_label_count : bool, optional
            If True, sort cluster assignment by total number of occurences of most
            occuring true label in cluster (fraction * cluster_size). Default is
            False, sorting cluster assignment by fraction of occurences.
        return_data_labels : bool, optional
            If True, return `data_labels`.
    
        Returns
        -------
        assigned_labels : ndarray
            If `return_data_labels` is True, a 1D array with size `data_size` with the assigned labels per
            data point is returned. Calling
    
        Examples
        --------
        >>> import numpy as np
        >>> data_cluster_indices = np.array([3, 3, 2, 1, 1, 1])
        >>> true_labels = np.array([1, 1, 2, 0, 2, 2])
        >>> res = ClusteringResults(data_cluster_indices)

        >>> assigned_labels = res.assign_cluster_labels_by_fraction(
        ...     true_labels, return_data_labels=True
        ... )
        >>> all(assigned_labels == res.data_labels)
        True
        >>> res.data_labels
        array([1, 1, 2, 0, 0, 0])
        >>> res.cluster_labels
        array([0, 2, 1])
    
        >>> res.assign_cluster_labels_by_fraction(true_labels,
        ...                                       distinct_labels=False)
        >>> res.data_labels
        array([1, 1, 2, 2, 2, 2])
        >>> res.cluster_labels
        array([2, 2, 1])
    
        >>> res.assign_cluster_labels_by_fraction(true_labels,
        ...                                       use_label_count=True)
        >>> res.data_labels
        array([1, 1, 0, 2, 2, 2])
        >>> res.cluster_labels
        array([2, 0, 1])
    
        '''
        if self.cluster_labels is not None:
            print("WARNING: Overwriting already existing cluster labels!")
        self.cluster_labels = - np.ones(self.num_clusters, dtype=int)

       # _, self.cluster_labels = assign_cluster_labels(
       #     self.data_indices, true_labels, distinct_labels=distinct_labels,
       #     use_label_count=use_label_count
       # )

        cluster_masks = []
        label_fractions = []
        highest_label_fractions = []
        unique_cluster_numbers = np.arange(1, self.num_clusters + 1)  # starts with 1
        for idx in unique_cluster_numbers:
            mask = (self.data_indices == idx)
            cluster_masks.append(mask)
            # get true labels for this cluster
            true_labels_per_cluster = true_labels[mask]
            # get the ocurrences of true labels in the cluster
            counts = np.bincount(true_labels_per_cluster, minlength=self.num_clusters)
            # compute their fraction to the total number of labels
            if use_label_count:
                # use total label counts instead of fractions
                fractions = counts
            else:
                fractions = counts / counts.sum()
            label_fractions.append(fractions)
            highest_label_fractions.append(fractions.max())
    
        if return_data_labels:
            # array of assigned label per data point (to be filled below)
            assigned_labels = - np.ones(self.data_size, dtype=int)
        # the new labels per cluster sorted by cluster indices (to be filled below)
        if distinct_labels:
            # assign labels in order of their "confidence" (highest_label_fraction)
            sort_idx = np.argsort(highest_label_fractions)[::-1]
            sorted_highest_label_fractions = list(np.array(highest_label_fractions)[sort_idx])
            sorted_unique_cluster_numbers = list(unique_cluster_numbers[sort_idx])
            used_labels = {}
            # sort label_options in descending order
            label_options_per_cluster = [list(fractions.argsort()[::-1]) for fractions in label_fractions]
            # outer loop through cluster indices
            num_iter_outer = len(sorted_unique_cluster_numbers)
            # offset in label_options array for each cluster (see below)
            idx_offsets_in_label_options = [0] * num_iter_outer
            for n, idx in enumerate(sorted_unique_cluster_numbers):
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
                        if return_data_labels:
                            # TODO assert all(assigned_labels[cluster_masks[idx]] == -1), "Trying to overassign labels"
                            assigned_labels[cluster_masks[idx]] = label
                        used_labels[label] = (idx, fraction)
                        self.cluster_labels[idx] = label
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
                        assert other_fraction >= fraction
                        if n == num_iter_outer - 1:
                            # last iteration of outer loop -> just loop through inner loop
                            continue
                        assert m < num_iter_inner - 1, \
                                'Finished inner loop without cluster assignment'
                        assert m<1000 and n<1000, 'Stuck in the loop? Bug?'
                        # make sure that the next assigned label is assigned to most confident
                        # cluster (highest_label_fraction)
                        next_idx = sorted_unique_cluster_numbers[n+1]
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
                            sorted_unique_cluster_numbers.insert(insert_position, idx+1)
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
        else:  # not distinct_labels
            # assign to each cluster the true label that is most represented
            for idx in unique_cluster_numbers:
                idx -= 1  # cluster indices start at 1
                label = label_fractions[idx].argmax()
                fraction = label_fractions[idx][label]  # max
                if return_data_labels:
                    assigned_labels[cluster_masks[idx]] = label
                self.cluster_labels[idx] = label
                print('Cluster {} gets label {} with fraction {:.3f}'.format(idx, label, fraction))
    
            print('Assigned labels: {}'.format(np.unique(self.cluster_labels)))
    
        if return_data_labels:
            assert not any(assigned_labels == -1), 'Not all labels were set! Assigned labels = {}'.format(np.unique(assigned_labels))
            return assigned_labels


def kmeans_clustering(X, num_clusters, seed=None, true_labels=None,
                       assign_clusters_with_label_count=False,
                       assign_labels_by_probability=False, distinct_labels=True):
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(X)
    cluster_results = ClusteringResults(kmeans.labels_ + 1,  # ClusteringResults expects cluster indices to start at 1
                                        cluster_centroids=kmeans.cluster_centers_,
                                        num_clusters=num_clusters)
    # TODO: kmeans.predict() is probably much faster then
    # ClusteringResults.classify_by_nearest_cluster()
    if true_labels is not None:
        if assign_labels_by_probability:
            cluster_results.assign_cluster_labels_by_closest_to_centroid(X, true_labels)
        else:
            cluster_results.assign_cluster_labels_by_fraction(
                true_labels, use_label_count=assign_clusters_with_label_count,
                distinct_labels=distinct_labels
            )

    return cluster_results


def hierarchical_clustering(X, linkage_method='ward', distance_metric='euclidean',
                            check_cophonetic=False, plot_dir=None,
                            truncate_dendrogram=None, num_clusters=None,
                            max_dist=None, true_labels=None,
                            assign_clusters_with_label_count=False,
                            assign_labels_by_probability=False, distinct_labels=True):
    '''
    hierarchical_clustering

    Parameters
    ----------
    X : ndarray
        Dataset in clustering space, shape (num_samples, num_dimensions).
    linkage_method : str, optional
        `scipy.cluster.hierarchy.linkage(..., method=linkage_method)`
    distance_metric : str, optional
        `scipy.cluster.hierarchy.linkage(..., metric=distance_metric)`
    check_cophonetic : bool, optional
        If True, prints Cophonetic Correlation Coefficient (memory intensive).
    plot_dir : str, optional
        Where to save dendrogram plot, default None (no plot created).
    truncate_dendrogram : int, optional
        How many merged clusters to show (None shows all).
    num_clusters : int, optional
        Determines where to cut the dendrogram for cluster assignment.
    max_dist : float, optional
        Determines where to cut dendrogram for cluster assignment.
    true_labels : ndarray, optional
        If given, cluster labels are chosen from highest occuring true_labels
        in cluster (default). If `assign_labels_by_probability` is `True`,
        cluster labels are chosen by label of data point with highest
        probability to be in the cluster.
    assign_clusters_with_label_count : bool, optional
        Use count of label occurences instead of fraction for assignments.
    assign_labels_by_probability : bool, optional
        If True, assign cluster labels by label of data point with highest
        probability to belong to that cluster.
    distinct_labels : bool, optional
        If True (default), enforce that all clusters are assigned distinct
        labels. If False, multiple clusters can have the same label.

    Returns
    -------
    ClusteringResults
        Class instance with attributes for cluster centroids and labels and
        cluster number and label assignment of the clustered data.
    '''
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
        data_cluster_indices = fcluster(Z, num_clusters, criterion='maxclust')

    if max_dist is not None:
        data_cluster_indices = fcluster(Z, max_dist, criterion='distance')
        num_clusters = int(np.max(data_cluster_indices))

    print('Cluster sizes:', np.bincount(data_cluster_indices, minlength=num_clusters+1))

    cluster_results = ClusteringResults(data_cluster_indices, data_coords=X,
                                        num_clusters=num_clusters)

    if true_labels is not None:
        if assign_labels_by_probability:
            cluster_results.assign_cluster_labels_by_closest_to_centroid(X, true_labels)
        else:
            cluster_results.assign_cluster_labels_by_fraction(
                true_labels, use_label_count=assign_clusters_with_label_count,
                distinct_labels=distinct_labels
            )

    return cluster_results
