import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from vae_tf.utils import random_subset, get_deconv_params
from vae_tf.mnist_helpers import load_mnist


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

def test_get_deconv_params():
    stride = (2, 2)
    filters = (5, 5)
    out_shape = (4, 4)
    in_shape = (1, 1)
    # should use 'VALID' padding since in > filter
    new_filter, new_stride, padding = get_deconv_params(out_shape, in_shape, filters, stride)
    assert_array_equal(new_filter, [4, 4])
    assert_array_equal(new_stride, [2, 2])
    assert_equal(padding, 'VALID')

    in_shape = (7, 7)
    out_shape = (14, 14)
    # should use 'SAME' padding since in < filter and out % in == 0
    new_filter, new_stride, padding = get_deconv_params(out_shape, in_shape, filters, stride)
    assert_array_equal(new_filter, [5, 5])
    assert_array_equal(new_stride, [2, 2])
    assert_equal(padding, 'SAME')

    in_shape = (7, 7)
    out_shape = (12, 12)
    # should use 'VALID' padding since out % in != 0 and decrease stride since otherwise s > k
    new_filter, new_stride, padding = get_deconv_params(out_shape, in_shape, filters, stride)
    assert_array_equal(new_filter, [6, 6])
    assert_array_equal(new_stride, [1, 1])
    assert_equal(padding, 'VALID')

    in_shape = (7, 1)
    out_shape = (14, 4)
    # should use 'VALID' padding since one dimension has in < filter
    new_filter, new_stride, padding = get_deconv_params(out_shape, in_shape, filters, stride)
    assert_array_equal(new_filter, [2, 4])
    assert_array_equal(new_stride, [2, 2])
    assert_equal(padding, 'VALID')

    in_shape = (7, 1)
    out_shape = (12, 4)
    # should use 'VALID' padding since one dimension has in < filter and decrease stride
    # in only that dimension since otherwise s > k, while not decreasing the other dims stride
    new_filter, new_stride, padding = get_deconv_params(out_shape, in_shape, filters, stride)
    assert_array_equal(new_filter, [6, 4])
    assert_array_equal(new_stride, [1, 2])
    assert_equal(padding, 'VALID')


if __name__ == '__main__':
    test_get_deconv_params()
