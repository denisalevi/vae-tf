import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from vae_tf.clustering import ClusteringResults

def test_clustering_results_class():
    data_cluster_indices = np.array([3, 3, 2, 1, 1, 1])
    true_labels = np.array([1, 1, 2, 0, 2, 2])
    cluster_labels = np.array([0, 2, 1])

    res = ClusteringResults(data_cluster_indices)

    assert_array_equal(res.data_labels, data_cluster_indices - 1)
    assert_raises(AssertionError, res.assign_cluster_labels_by_closest_to_centroid,
                  np.random.randint(10, size=data_cluster_indices.shape), true_labels)
    assert_raises(AssertionError, res.classify_by_nearest_cluster,
                  np.random.randint(10, size=2))
    assert_array_equal(res.num_clusters, 3)
    assert_array_equal(res.data_size, 6)

    data_coords = np.array([9, 7, 5, 3, 2, 1]).reshape((6, 1))
    expected_centroids = np.array([2, 5, 8]).reshape((3, 1))
    expected_cluster_labels_clostest = np.array([2, 2, 1])
    expected_data_labels = np.array([1, 1, 2, 2, 2, 2])
    classify_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((11, 1))
    expected_classify_data_cluster_number = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    expected_classify_data_labels = np.array([2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1])

    res.compute_cluster_centroids(data_coords)
    assert_array_equal(res.cluster_centroids, expected_centroids)

    res2 = ClusteringResults(data_cluster_indices, data_coords=data_coords)
    assert_array_equal(res.cluster_centroids, res2.cluster_centroids)

    assert_raises(AssertionError, res.assign_cluster_labels_by_closest_to_centroid,
                  np.random.randint(10, size=5), true_labels)

    # no cluster labels yet, we should get cluster_number - 1 from data_labels
    assert_array_equal(res.data_labels, res.data_indices - 1)
    assert_array_equal(res.classify_by_nearest_cluster(classify_data),
                       expected_classify_data_cluster_number - 1)

    # 1D data input array
    assert_raises(AssertionError, res.assign_cluster_labels_by_closest_to_centroid,
                  np.random.randint(10, size=10), true_labels)
    # assign cluster labels
    res.assign_cluster_labels_by_closest_to_centroid(data_coords, true_labels)
    assert_array_equal(res.cluster_labels, expected_cluster_labels_clostest)
    assert_array_equal(res.data_labels, expected_data_labels)
    assert_array_equal(res.classify_by_nearest_cluster(classify_data),
                       expected_classify_data_labels)

def test_assign_cluster_labels_by_fraction():
    data_cluster_indices = np.array([3, 3, 2, 1, 1, 1])
    true_labels = np.array([1, 1, 2, 0, 2, 2])

    res = ClusteringResults(data_cluster_indices)
    assigned_labels = res.assign_cluster_labels_by_fraction(true_labels,
                                                            return_data_labels=True)
    assert_array_equal(res.data_labels, assigned_labels)
    assert_array_equal(res.data_labels, [1, 1, 2, 0, 0, 0])
    assert_array_equal(res.cluster_labels, [0, 2, 1])

    res.assign_cluster_labels_by_fraction(true_labels, distinct_labels=False)
    assert_array_equal(res.data_labels, [1, 1, 2, 2, 2, 2])
    assert_array_equal(res.cluster_labels, [2, 2, 1])

    res.assign_cluster_labels_by_fraction(true_labels, use_label_count=True)
    assert_array_equal(res.data_labels, [1, 1, 0, 2, 2, 2])
    assert_array_equal(res.cluster_labels, [2, 0, 1])

def test_assign_cluster_labels_by_fraction_bug():
    "Test for a bug that occured when a cluster had no members of the highest label option"
    # this reproduces the bug
    # cluster idx 1 will be assigned a label last, after all its label options (0, 1)
    # are already assigned
    data_cluster_indices = np.array([1,1,1,1,
                                2,2,2,2,
                                3,3,3,3])

    true_labels = np.array([0,0,0,0,
                            0,0,0,1,
                            1,1,1,1])
        
    res = ClusteringResults(data_cluster_indices)
    res.assign_cluster_labels_by_fraction(true_labels)

    expected_labels = np.array([0,0,0,0,
                                2,2,2,2,
                                1,1,1,1])

    assert_array_equal(res.data_labels, expected_labels)


if __name__ == '__main__':
    test_assign_cluster_labels_by_fraction()
    test_clustering_results_class()
    test_assign_cluster_labels_by_fraction_bug()
