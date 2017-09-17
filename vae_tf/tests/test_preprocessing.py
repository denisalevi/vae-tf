import os
import pandas as pd
import numpy as np
import pickle
import time
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt

from vae_tf.bout_helpers.datasets import extract_bouts_and_labels, load_bouts
from vae_tf.bout_helpers.preprocessing import (rescale_data, standardize_by_entire_dataset,
                                               standardize_per_data_point,
                                               standardize_per_fragment_dimension,
                                               calculate_reverse_transform)

# choose some small dataset for testing
DATADIR = '/home/denisalevi/projects/deep_learning_champalimaud/data/minExampleBoutFiles/boutFilesFinal_npz/one_exp_padded_300/bout_data.npz'

def test_standardize_by_entire_data():

    bouts = load_bouts(DATADIR).train.bouts.astype('float32')
    original = bouts.copy()

    scale, shift = standardize_by_entire_dataset(bouts, return_transforms=True)
    assert_allclose(bouts.mean(), 0, atol=1e-5)
    assert_allclose(bouts.std(), 1, atol=1e-5)
    assert_allclose(bouts, original * scale + shift, atol=1e-5)
    assert_allclose(original.mean(), (bouts - shift / scale).mean(), atol=1e-5)
    assert_allclose(original.std(), (bouts / scale).std(), atol=1e-5)

def test_standardze_per_data_point():

    bouts = load_bouts(DATADIR).train.bouts.astype('float32')
    original = bouts.copy()

    scales, shifts = standardize_per_data_point(bouts, return_transforms=True)
    assert_allclose(bouts.mean(axis=(1, 2, 3)), 0, atol=1e-5)
    assert_allclose(bouts.std(axis=(1, 2, 3)), 1, atol=1e-3)
    assert_allclose(bouts, original * scales + shifts, atol=1e-5)
    assert_allclose(original.mean(axis=(1, 2, 3)), (bouts - shifts / scales).mean(axis=(1, 2, 3)), atol=1e-3)
    assert_allclose(original.std(axis=(1, 2, 3)), (bouts / scales).std(axis=(1, 2, 3)), atol=1e-3)

def test_standardize_per_fragment_dimension():

    bouts = load_bouts(DATADIR).train.bouts.astype('float32')
    original = bouts.copy()

    scales, shifts = standardize_per_fragment_dimension(bouts, return_transforms=True)
    assert_allclose(bouts.mean(axis=(0, 2, 3)), 0, atol=1e-5)
    assert_allclose(bouts.std(axis=(0, 2, 3)), 1, atol=1e-5)
    assert_allclose(bouts, original * scales + shifts, atol=1e-5)
    assert_allclose(original.mean(axis=(0, 2, 3)), (bouts - shifts / scales).mean(axis=(0, 2, 3)), atol=1e-3)
    assert_allclose(original.std(axis=(0, 2, 3)), (bouts / scales).std(axis=(0, 2, 3)), atol=1e-3)

def test_rescaling_data():

    bouts = load_bouts(DATADIR).train.bouts.astype('float32')
    original = bouts.copy()

    scale, shift = rescale_data(bouts, return_transforms=True)
    assert_allclose(bouts.min(), 0, atol=1e-5)
    assert_allclose(bouts.max(), 1, atol=1e-5)
    assert_allclose(bouts, original * scale + shift, atol=1e-5)

    original = bouts.copy()
    scale, shift = rescale_data(bouts, low=-2, high=2, return_transforms=True)
    assert_allclose(bouts.min(), -2, atol=1e-5)
    assert_allclose(bouts.max(), 2, atol=1e-5)
    assert_allclose(bouts, original * scale + shift, atol=1e-5)

def test_calculate_reverse_transform():

    bouts = load_bouts(DATADIR).train.bouts.astype('float32')
    original = bouts.copy()

    scale, shift = standardize_by_entire_dataset(bouts, return_transforms=True)
    scale2, shift2 = rescale_data(bouts, return_transforms=True)
    rev_scale, rev_shift = calculate_reverse_transform([scale, scale2], [shift, shift2])
    assert_allclose(original, bouts * rev_scale + rev_shift, atol=1e-3)

def test_extract_bouts_and_labels():

    filename = '/tmp/test_bouts.npz'

    bout_frame_sizes = np.array([[1, 2, 3], [3, 1, 2], [3, 3, 2]])
    original_bout_len = bout_frame_sizes * 8 + 2
    assert np.all(original_bout_len == np.array([[10, 18, 26], [26, 10, 18], [26, 26, 18]]))
    original_start_idx = original_bout_len.cumsum(axis=1).astype('int32')
    assert np.all(original_start_idx == np.array([[10, 28, 54], [26, 36, 54], [26, 52, 70]]))

    original_bouts = [np.random.randint(low=np.iinfo(np.int16).min,
                                        high=np.iinfo(np.int16).max + 1,
                                        size=n, dtype=np.int16)
                      for n in original_start_idx[:, -1]]
    # exp ids
    original_bouts[0][0] = 0
    original_bouts[1][0] = 1
    original_bouts[2][0] = 2
    original_bouts[0][original_start_idx[0, :-1]] = 0
    original_bouts[1][original_start_idx[1, :-1]] = 1
    original_bouts[2][original_start_idx[2, :-1]] = 2

    np.savez(filename, experiment_id_name_lookup={'name_to_id':{'exp0':0, 'exp1':1, 'exp2':2}},
             int16_to_rad_factor=1,
             exp0=original_bouts[0], exp0_start_indices=original_start_idx[0],
             exp1=original_bouts[1], exp1_start_indices=original_start_idx[1],
             exp2=original_bouts[2], exp2_start_indices=original_start_idx[2])


    with np.load(filename) as f:
        #experiments = ['3minLightDark2']
        #experiments = ['3minLightDark2', 'Beeps200to2000', 'BeepsLightDarkEyeConv']
        experiments = 'all'
        max_bout_len = 100#1600#300

        bouts, mar, exp, start_idx = extract_bouts_and_labels(f, experiments, max_bout_len, padding=False,
                                                              create_df=False, bout_dtype=np.float32,
                                                              pad_value=np.nan, testing=True)

        # testing = True
        bouts2, mar2, exp2 = extract_bouts_and_labels(f, experiments, 2, padding=True,
                                                              create_df=False, bout_dtype=np.float32,
                                                              pad_value=0, testing=True)
        assert bouts2.shape[0] == 5, bouts2.shape

        # padding with np.nan and no testing
        bouts2, mar2, exp2 = extract_bouts_and_labels(f, experiments, max_bout_len, padding=True,
                                                              create_df=False, bout_dtype=np.float32,
                                                              pad_value=np.nan, testing=False)

    assert len(mar) == len(exp) == len(start_idx), "{}\t{}\t{}".format(len(mar), len(exp), len(start_idx))
    assert np.all(start_idx[1:] >= start_idx[:-1])  # monotonicity
    assert start_idx[-1] == bouts.shape[2], "last end idx {}\tbout frames {}".format(start_idx[-5:], bouts.shape[2])

    bout_split = np.split(bouts, start_idx[:-1], axis=2)
    #print('last few splits', bout_split[-5:], 'last shape', bout_split[-1].shape)
    assert len(bout_split) == bouts2.shape[0]
    new_bout_list = []
    for i, bout in enumerate(bout_split):
        if bout_split[i].size == 0:
            break
    for i, bout in enumerate(bout_split):
        assert bout.ndim == 4
        assert bout.shape[0] == 1
        assert bout.shape[1] == 8 
        assert bout.shape[3] == 1
        pad_len = max_bout_len - bout.shape[2]
        if pad_len < 0:
            print('i bout shape', i, bout.shape)
        bout = np.lib.pad(bout,
                          pad_width=((0, 0), (0, 0), (0, pad_len), (0, 0)),
                          mode='constant',
                          constant_values=(np.nan,))
        assert bout.shape[2] == max_bout_len
        new_bout_list.append(bout)

    bouts_rejoined = np.vstack(new_bout_list)
    assert bouts_rejoined.shape == bouts2.shape, "rejoin {} correct {}".format(bouts_rejoined.shape, bouts2.shape)
    assert_array_equal(bouts_rejoined, bouts2)


if __name__ == '__main__':
    #test_extract_bouts_and_labels()
    #test_standardize_by_entire_data()
    #test_standardze_per_data_point()
    #test_standardize_per_fragment_dimension()
    #test_rescaling_data()
    test_calculate_reverse_transform()
