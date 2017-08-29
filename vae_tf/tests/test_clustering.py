import numpy as np
from numpy.testing import assert_array_equal

from vae_tf.clustering import assign_cluster_labels


def test_assign_cluster_labels():
    "Test for a bug that occured when a cluster had no members of the highest label option"
    # this reproduces the bug
    # cluster idx 1 will be assigned a label last, after all its label options (0, 1)
    # are already assigned
    cluster_indices = np.array([1,1,1,1,
                                2,2,2,2,
                                3,3,3,3])

    true_labels = np.array([0,0,0,0,
                            0,0,0,1,
                            1,1,1,1])
        
    cluster_labels = assign_cluster_labels(cluster_indices, true_labels)

    expected_cluster_labels = np.array([0,0,0,0,
                                        2,2,2,2,
                                        1,1,1,1])

    assert_array_equal(cluster_labels, expected_cluster_labels)


if __name__ == '__main__':
    test_assign_cluster_labels()


