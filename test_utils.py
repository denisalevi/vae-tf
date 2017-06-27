import numpy as np
from numpy.testing import assert_equal

from utils import random_subset
from main import load_mnist


def test_random_subset():
    mnist = load_mnist()
    images = mnist.test.images
    labels = mnist.test.labels

    # basic test
    im1, _ = random_subset(images, 100, same_num_labels=False)
    assert_equal(im1.shape[0], 100)

    # basic test with labels
    _, lab1 = random_subset(images, 100, labels=labels, same_num_labels=False)
    assert_equal(lab1.shape[0], 100)

    # test same_num_labels
    im2, lab2 = random_subset(images, 100, labels=labels, same_num_labels=True)
    for label in np.unique(lab2):
        assert_equal(lab2[lab2 == label].shape[0], 10)

    # test same_num_labels for undividable size
    im3, lab3 = random_subset(images, 101, labels=labels, same_num_labels=True)
    sizes = []
    for label in np.unique(lab3):
        size = lab3[lab3 == label].shape[0]
        sizes.append(size)
    sizes = np.array(sizes)
    assert_equal(sizes[sizes == 11].size ,1)
    assert_equal(sizes[sizes == 10].size ,9)

    # test additional labels
    add_labels = np.random.randint(0, high=10, size=(mnist.test.labels.shape[0]))
    double_labels = np.vstack([labels, add_labels]).T
    im4, lab4 = random_subset(images, 100, labels=double_labels)
    assert_equal(lab4.shape, (100, 2))

if __name__ == '__main__':
    test_random_subset()
