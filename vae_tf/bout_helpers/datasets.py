# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading bout data."""

import random
import os
import time

import numpy as np
import pandas as pd
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_bouts_and_labels(f, experiments, max_bout_len, padding, create_df=False,
                             max_bout_angle=None, discard_large_angles=None,
                             max_tail_frags=None,
                             bout_dtype=np.int16, pad_value=0, testing=False):
    '''
    Extract the bout data and labels into 4D and 1D arrays.

    Parameters
    ----------
    f : NpZFile object
        `.npz` file laoded with numpy.load(), see `read_data_sets` for detail.
    experiments : {"all", list(experiment_names)}
        See `read_data_sets` doc string.
    padding : bool
        If True, pad each bout with 0 in the frameIdx axis until all bouts have
        the same dimensions. If `max_bout_len` is "max_experiments", all
        bouts will have the dimensions of the longest bout in the specified
        experiment subset (`experiments` argument). If `max_bout_len` is
        "max_data", all bouts will have the dimensions of the longest
        bout in the entire `data_archive`. If `padding` is False, all bouts are
        concatenated along the frameIdx axis and the start indices for
        splitting them are returned.
    max_bout_len : int or str
        Integer or one of {"max_experiments", "max_data"}. See `read_data_sets`
        doc string for details.
    max_bout_angle : float, optional
        Modify bouts where the highest angle exceeds `max_bout_angle`. The
        `discard_large_angles` option specifies what to do.
    discard_large_angles : {"delete_fragments", "delete_bout"}, optional
        What to do with bouts that have angles exceeding `max_bout_angle`. If
        "delete_fragments", delete tail fragment data (set to 0) starting from
        the lowest fragement that exceeds `max_bout_angle`. If "delete_bout",
        discard the entire bout.
    max_tail_frags : int, optional
        How many tail fragments to save in dataset. Maximum is 8 (all).
    bout_dtype : {np.int16, np.float32}
        The datatype of the bout data. If np.float32, the data is rescaled into
        radian.
    pad_value : int or float, optional
        The value to pad the bouts with.
    testing : bool, optional
        If True, create arrays with set "invalidy' value for all generated data
        and check that all array positions where set. If False (default), create
        np.empty arrays (values as are found in memory), which is faster but
        cannot be checked for unset values.

    Returns
    -------
    bouts : ndarray
        4D array of 16 bit integers of shape (num_bouts, num_tail_fragments,
        num_frames, 1) with tail fragment angles for bout movements
        (represented in [-32767, 32767] integers or in radian floats, depending
        on `bout_dtype` argument). If padding is False, all bouts are
        concatenated along the frameIdx dimension.
    marques_labels : ndarray
        1D array of 32 bit integers of lengths `num_bouts` with marques
        classification IDs.
    experiment_ids : ndarray
        1D array of 32 bit integers of lengths `num_bouts` with experiment IDs.
    bout_start_indices : ndarray
        Only returned if padding is False. 1D array of 32 bit integers of
        length `num_bouts` specifying the start indices of the bouts along the
        frameIdx dimension. First element is the start idx of the second bout.
        Last element is the total number of frames + 1. For np.split use 
        `bout_start_indices[:-1]` as split indices.
    '''
    lookup = f['experiment_id_name_lookup'].item()
    int16_to_rad_factor = float(f['int16_to_rad_factor'])
    if experiments == 'all':
        #experiments = [exp for exp in lookup['name_to_id'].keys() if exp != 'BeepsLightDarkEyeConv']
        experiments = lookup['name_to_id'].keys()
    else:
        if not isinstance(experiments, list):
            raise ValueError("`experiments` needs to be 'all' or of type list, got {}".
                             format(type(experiments)))

    if not padding and create_df:
        raise ValueError("Can't create DataFrame if `padding` is False")

    if bout_dtype not in [np.int16, np.float32]:
        raise ValueError('`bout_dtype` has to be `np.int16` or `np.float32`')

    if np.isnan(pad_value) and bout_dtype != np.float32:
        raise ValueError("Can't pad with `np.nan` if `bout_dtype` is not a floating point")

    if bout_dtype == np.float32 and testing and padding and np.isnan(pad_value):
        raise ValueError("Can't be testing (which uses np.nan) when padding with np.nan")

    if bout_dtype == np.int16 and testing and padding and pad_value == np.iinfo(np.int16):
        raise ValueError("`pad_value` is the same as 'invalidity' value for testing")

    find_num_frames = None
    if max_bout_len in ['max_experiments', 'max_data']:
        find_num_frames = max_bout_len
        max_bout_len = 0
    elif not isinstance(max_bout_len, int):
        raise ValueError('`max_bout_len` has to be `int` or one of {"max_experiments", '
                         '"max_data"}, got {}'.format(max_bout_len))

    if discard_large_angles is not None and discard_large_angles not in ['delete_fragments', 'delete_bout']:
        raise ValueError('`discard_large_angles` can only be one of ["delete_fragments", "delete_bout"]')

    # dataset recorded 8 tail fragments
    num_tail_fragments = 8

    if max_tail_frags > num_tail_fragments:
        raise ValueError("`max_tail_frags` ({}) is larger then available tail fragments ({})"
                         .format(max_tail_frags, num_tail_fragments))

    # LOAD BOUT START INDICES FIRST TO GET TOTAL NUMBER OF BOUTS AND MAX BOUT LENGTH
    loaded_exp_start_idx = {}  # pointers to loaded bout start indices
    exp_max_bout_lengths = {}
    exp_max_bout_angles = {}
    num_bouts = 0
    if not padding:
        total_num_frames = 0
    for exp in experiments:
        if exp not in f.keys():
            raise ValueError('Experiment "{}" not in data'.format(exp))
        exp_start_idx = f[exp + '_start_indices']
        assert exp_start_idx.dtype == np.int32, exp_start_idx.dtype
        loaded_exp_start_idx[exp] = exp_start_idx
        num_bouts_this_exp = len(exp_start_idx)
        num_bouts += num_bouts_this_exp
        if find_num_frames is not None or not padding:
            # add first bout idx to bout indices
            bout_lengths = np.diff(np.append([0], exp_start_idx))
            assert len(bout_lengths) == num_bouts_this_exp
            if find_num_frames is not None:
                # the flat bout data consists of 2 labels and num_tail_fragments * num_frames angle values
                num_angle_values = int(bout_lengths.max()) - 2
                assert num_angle_values % num_tail_fragments == 0
                max_bout_len_this_exp = num_angle_values // num_tail_fragments
                if max_bout_len_this_exp > max_bout_len:
                    max_bout_len = max_bout_len_this_exp
            if not padding:
                total_num_angle_values = int(bout_lengths.sum()) - len(bout_lengths) * 2
                assert total_num_angle_values % num_tail_fragments == 0
                total_num_frames += total_num_angle_values // num_tail_fragments

    if find_num_frames == 'max_data':
        print('ALL DATA ARCHIVE')
        # find the max bout length in the entire data archive
        # loop through the experiments we didn't loop through before
        for exp in lookup['name_to_id'].keys() - experiments:  # difference
            exp_start_idx = f[exp + '_start_indices']
            # the flat bout data consists of 2 labels and num_tail_fragments * num_frames angle values
            num_angle_values = int(np.diff(np.append([0], exp_start_idx)).max()) - 2
            assert num_angle_values % num_tail_fragments == 0
            max_bout_len_this_exp = num_angle_values // num_tail_fragments
            if max_bout_len_this_exp > max_bout_len:
                max_bout_len = max_bout_len_this_exp

    # PREPARE NUMPY ARRAY FOR BOUT DATA
    print('INFO total num bouts: {}, number of frames: {}, bout length: {}'
          .format(num_bouts, max_tail_frags, max_bout_len))
    num_bytes = 2 if bout_dtype == np.int16 else 4
    if padding:
        num_angles = num_bouts * max_bout_len * max_tail_frags
    else:
        num_angles = total_num_frames * max_tail_frags 
    print('INFO number of values ({}) in bouts array: {} ({:.2f} GB memory)'
          .format(bout_dtype.__name__, num_angles, num_angles * num_bytes / 1024 / 1024 / 1024))

    if testing:
        # create arrays filled with some invalid value to check later that all values were set
        # invalid values for testing
        int32_invalid_value = np.iinfo(np.int32).min
        if bout_dtype == np.int16:
            bout_invalid_val = np.iinfo(np.int16).min
        elif bout_dtype == np.float32:
            bout_invalid_val = np.nan

        # allocate arrays
        if not padding:
            # create array to store per tail fragment all frames of all bouts in one long array
            bouts = bout_invalid_val * np.ones((1, max_tail_frags, total_num_frames, 1), dtype=bout_dtype)
            # index going through all frames (total_num_frames)
            bout_start_indices = int32_invalid_value * np.ones(num_bouts, dtype=np.int32)
        elif not create_df:
            bouts = bout_invalid_val * np.ones((num_bouts, max_tail_frags, max_bout_len, 1), dtype=bout_dtype)
        marques_labels = int32_invalid_value * np.ones(num_bouts, dtype=np.int32)
        experiment_ids = int32_invalid_value * np.ones(num_bouts, dtype=np.int32)
    else:  # not testing
        # create empty arrays (better performance)
        if not padding:
            # create empty array to store per tail fragment all frames of all bouts in one long array
            bouts = np.empty((1, max_tail_frags, total_num_frames, 1), dtype=bout_dtype)
            # index going through all frames (total_num_frames)
            bout_start_indices = np.empty(num_bouts, dtype=np.int32)
        elif not create_df:
            bouts = np.empty((num_bouts, max_tail_frags, max_bout_len, 1), dtype=bout_dtype)
        marques_labels = np.empty(num_bouts, dtype=np.int32)
        experiment_ids = np.empty(num_bouts, dtype=np.int32)

    bout_idx = 0
    num_deleted_bouts = 0
    if not padding:
        frame_idx = 0
        this_exp_start_frame = 0
        num_deleted_frames = 0
    modified_bout_indices = {'max_len' : [], 'max_angle' : []}
    # LOAD BOUT DATA
    exp_dfs = {}
    bout_idx_this_exp = None
    for i_exp, exp in enumerate(experiments):
        print("Loading experiment {}/{} {}".format(i_exp+1, len(experiments), exp))
        exp_data = f[exp]
        bout_idx_this_exp = 0
        assert exp_data.dtype == np.int16, exp_data.dtype
        if bout_dtype == np.float32:
            exp_data = exp_data.astype(np.float32)
        exp_start_idx = loaded_exp_start_idx[exp]
        exp_id = lookup['name_to_id'][exp]
        # split flat data array into different bouts (first start idx is 0)
        bout_data = np.split(exp_data, exp_start_idx[:-1])
        bout_dfs = {}
        if not padding:
            # exp_start_idx gives indices in flat array from data archive
            # prepare exp_start_idx for start index in reshaped bouts array
            # substract 2 indices per bout for marques label and experiment id
            subtract = np.arange(1, len(exp_start_idx) + 1) * 2
            exp_start_idx -= subtract
            # devide by the number of tail fragments
            assert np.all(np.mod(exp_start_idx, num_tail_fragments) == 0), exp_start_idx[:5]
            exp_start_idx //= num_tail_fragments
        for i_bout, bout in enumerate(bout_data):
            # extract bout infos and data
            bout_exp_id = bout[0]  # see readme for experiment names
            assert bout_exp_id == exp_id
            marques_label = bout[1]  # Marques classification labels
            # we have 8 angles per frame, -1 infers number of frames from array len
            bout_frames = bout[2:].reshape((num_tail_fragments, -1))
            if max_tail_frags < num_tail_fragments:
                bout_frames = bout_frames[:max_tail_frags + 1, :]
            if not padding:
                if bout_idx_this_exp != 0:
                    assert bout_frames.shape[1] == exp_start_idx[bout_idx_this_exp] - exp_start_idx[bout_idx_this_exp-1]
                else:
                    assert bout_frames.shape[1] == exp_start_idx[bout_idx_this_exp]
            if bout_frames.shape[1] > max_bout_len:
                # store the unique (experimentId, boutIdx) tuple
                modified_bout_indices['max_len'].append((exp_id, i_bout))
                num_deleted_bouts += 1
                if not padding:
                    num_deleted_frames += bout_frames.shape[1]
                    # reduce the start indices of the following bouts by the lengths of this bout
                    # the exp_start_idx start with idx for the second bout (skipping the 0 for the first)
                    exp_start_idx[bout_idx_this_exp:] -= bout_frames.shape[1]
                    assert exp_start_idx[bout_idx_this_exp] == exp_start_idx[bout_idx_this_exp - 1]
                    if bout_idx_this_exp != 0:
                        # delete the start index of this bout
                        exp_start_idx = np.delete(exp_start_idx, bout_idx_this_exp - 1)
                # skip this bout
                continue

            if discard_large_angles is not None:
                if discard_large_angles == 'delete_bout':
                    max_angle = bout_frames.max()
                    if max_angle * int16_to_rad_factor > max_bout_angle:
                        modified_bout_indices['max_angle'].append((exp_id, i_bout))
                        num_deleted_bouts += 1
                        if not padding:
                            num_deleted_frames += bout_frames.shape[1]
                            # reduce the start indices of the following bouts by the lengths of this bout
                            # the exp_start_idx start with idx for the second bout (skipping the 0 for the first)
                            exp_start_idx[bout_idx_this_exp:] -= bout_frames.shape[1]
                            assert exp_start_idx[bout_idx_this_exp] == exp_start_idx[bout_idx_this_exp - 1]
                            if bout_idx_this_exp != 0:
                                # delete the start index of this bout
                                exp_start_idx = np.delete(exp_start_idx, bout_idx_this_exp - 1)
                        # skip this bout
                        continue

                elif discard_large_angles == 'delete_fragments':
                    max_angle_idx = np.unravel(bout_frames.argmax(), bout_frames.shape)
                    max_angle = bout_frames[max_angle_idx]
                    if max_angle * int16_to_rad_factor > max_bout_angle:
                        modified_bout_indices['max_angle'].append((exp_id, i_bout))
                        max_angle_fragment = max_angle_idx[0]
                        if max_angle_fragment < 7:
                            print('WARNING: max angle found in fragment {} for experimentId {} '
                                  'and boutIdx {}'.format(max_angle_fragment, exp_id, i_bout))
                        # set angles from tail fragment with highest angle on to 0
                        bout_frames[max_angle_fragment:, :] = 0

            if padding:
                # pad end of each row with 0 to get max_bout_len frames
                pad_len = max_bout_len - bout_frames.shape[1]
                # TODO for padding == False we could mask the padding and cut it out later
                bout_frames = np.lib.pad(bout_frames,
                                         pad_width=((0, 0), (0, pad_len)),
                                         mode='constant',
                                         constant_values=(pad_value,))
                assert bout_frames.shape[1] == max_bout_len

                if create_df:
                    bout_dfs[i_bout] = pd.DataFrame(bout_frames.T, dtype=bout_dtype)
                else:
                    assert bouts[bout_idx, ...].shape == (num_tail_fragments, max_bout_len, 1)
                    if testing:
                        # make sure the values we set wer not set before
                        if np.isnan(bout_invalid_val):
                            assert np.all(np.isnan(bouts[bout_idx, :, :, 0])), 'overwriting previously set value'
                        else:
                            assert np.all(bouts[bout_idx, :, :, 0] == bout_invalid_val), 'overwriting previously set value'
                    bouts[bout_idx, :, :, 0] = bout_frames
            else:  # no padding
                frame_length = bout_frames.shape[1]
                start = frame_idx
                end = frame_idx + frame_length
                if testing:
                    if np.isnan(bout_invalid_val):
                        assert np.all(np.isnan(bouts[0, :, start:end, 0]))
                    else:
                        assert np.all(bouts[0, :, start:end, 0] == bout_invalid_val)
                bouts[0, :, start:end, 0] = bout_frames
                frame_idx += frame_length
                bout_idx_this_exp += 1
            if testing:
                # make sure the values we set wer not set before
                assert np.all(marques_labels[bout_idx] == int32_invalid_value), 'overwriting previously set value'
                assert np.all(experiment_ids[bout_idx] == int32_invalid_value), 'overwriting previously set value'
            marques_labels[bout_idx] = marques_label
            experiment_ids[bout_idx] = exp_id
            bout_idx += 1

        if not padding:
            num_bouts_this_exp = len(exp_start_idx)
            end = bout_idx
            start = bout_idx - num_bouts_this_exp
            if testing:
                # make sure the values we set wer not set before
                assert np.all(bout_start_indices[start:end] == int32_invalid_value), 'overwriting previously set value'
            # exp_start_idx is counting frames from 0 per bout
            bout_start_indices[start:end] = exp_start_idx + this_exp_start_frame
            this_exp_start_frame = frame_idx
        elif create_df:
            exp_dfs[exp_id] = pd.concat(bout_dfs)
            del bout_dfs

    # store the bouts that were deleted / modified as pandas DataFrame
    # TODO write to file
    arr = []
    for typ, indices in modified_bout_indices.items(): 
        for idx in indices:
            arr.append([typ, *idx])
    df = pd.DataFrame(arr, columns=['type', 'experimentId', 'boutIdx'])
    print('MODIFIED BOUTS\n', df)

    if find_num_frames is None:
        print("INFO: skipped {} bouts longer then {} frames and skipped/modified {} bouts with angles "
              "larger then {} rad out of a total of {} bouts.".format(
                  len(modified_bout_indices['max_len']), max_bout_len,
                  len(modified_bout_indices['max_angle']), max_bout_angle, num_bouts))
    # TODO this was wrong? 'max_angle' can also be filled when fragments are set to 0 but not deleted?
    #num_deleted_bouts = len(modified_bout_indices['max_len']) + len(modified_bout_indices['max_angle'])

    if create_df:
        bouts = pd.concat(exp_dfs)
        bouts.index.names = ['experimentId', 'boutIdx', 'frameIdx']
        bouts.columns.names = ['tailFragmentIdx']
        dtype = bouts.values.dtype
        bout_values = bouts.values
    elif padding:
        if num_deleted_bouts != 0:
            bouts = bouts[:-num_deleted_bouts]
        dtype = bouts.dtype
        bout_values = bouts
    else:  # no padding
        if num_deleted_frames != 0:
            bouts = bouts[:, :, :-num_deleted_frames, :]
        if num_deleted_bouts != 0:
            bout_start_indices = bout_start_indices[:-num_deleted_bouts]
        assert bouts.shape[2] == total_num_frames - num_deleted_frames
        bout_values = bouts

    if num_deleted_bouts != 0:
        marques_labels = marques_labels[:-num_deleted_bouts]
        experiment_ids = experiment_ids[:-num_deleted_bouts]

    if testing:
        if np.isnan(bout_invalid_val):
            assert not np.any(np.isnan(bout_values)), "{} NaNs shape {} (bouts shape {}):\n{}".format(len(np.argwhere(np.isnan(bout_values))),
                                                                                                      np.argwhere(np.isnan(bout_values)).shape, 
                                                                                                      bout_values.shape,
                                                                                                      np.argwhere(np.isnan(bout_values)))
        else:
            assert not np.any(bout_values == bout_invalid_val)

        if not padding:
            assert not np.any(bout_start_indices == int32_invalid_value)

        assert not np.any(marques_labels == int32_invalid_value)
        assert not np.any(experiment_ids == int32_invalid_value)
        print('Testing passed, no "invalid" array values!')

    if bout_dtype == np.float32:
        bouts *= int16_to_rad_factor

    if padding:
        return bouts, marques_labels, experiment_ids
    else:
        return bouts, marques_labels, experiment_ids, bout_start_indices


class DataSet(object):

    def __init__(self,
                 bouts,
                 experiment_ids=None,
                 marques_labels=None,
                 fake_data=False,
                 dtype=dtypes.int16,
                 int16_to_rad_factor=None,
                 reshape=True,
                 marques_label_name_lookup=None,
                 experiment_id_name_lookup=None):
        """Construct a DataSet.
        `dtype` can be either `int16` to leave the input as `[-32,767,
        32,767]`, or `float32` to rescale into radian with the
        `int16_to_rad_factor` argument.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.int16, dtypes.float32):
            raise TypeError('Invalid bout dtype %r, expected int16 or float32' %
                            dtype)
        if dtype == dtypes.float32 and int16_to_rad_factor is None:
            raise ValueError('For 32 bit float data, `int16_to_rad_factor` argument '
                             'is needed')

        self.int16_to_rad_factor = int16_to_rad_factor
        self.marques_label_name_lookup = marques_label_name_lookup
        self.experiment_id_name_lookup = experiment_id_name_lookup

        if fake_data:
            # create one batch of random images
            bouts = np.random.randint(-32767, 32767, size=(1, 8, 300, 1))
            marques_labels = np.array([0])
            experiment_ids = np.array([0])
        else:
            assert bouts.shape[0] == marques_labels.shape[0] == experiment_ids.shape[0],\
                    'bouts.shape: {} marques_labels.shape: {} experiment_ids.shape: {}'\
                    .format(bouts.shape, marques_labels.shape, experiment_ids.shape)


            if reshape:
                # Convert shape from [num examples, rows, columns, depth]
                # to [num examples, rows*columns] (assuming depth == 1)
                assert bouts.shape[3] == 1
                bouts = bouts.reshape(bouts.shape[0], bouts.shape[1] * bouts.shape[2])
            if dtype == dtypes.float32:
                # Convert from [-32,767, 32,767] -> [0.0, MAX_RAD].
                # np.float32 has 23 bit mantissa and 8 bits exponent (fits np.int16)
                bouts = bouts.astype(dtypes.float32)
                bouts = np.multiply(bouts, int16_to_rad_factor)

        self._num_examples = bouts.shape[0]
        self._bout_dims = tuple(bouts.shape[1:3])
        self._bouts = bouts
        self._marques_labels = marques_labels  #:ndarray label IDs
        self._experiment_ids = experiment_ids  #:ndarray experiment IDs
        self._epochs_completed = 0
        self._index_in_epoch = 0

    # leave images property for compatibility
    @property
    def images(self):
        raise AttributeError('bout DataSet has no `images` attribute. Use '
                             '`bouts` instead.')
        print("INFO: Using DataSet.images for bout data!")
        return self._bouts

    @property
    def labels(self):
        raise AttributeError('bout DataSet has no `labels` attribute. Use '
                             '`marques_labels` or `experiment_ids`.')

    @property
    def bouts(self):
        return self._bouts

    @property
    def bout_dims(self):
        return self._bout_dims

    @property
    def marques_labels(self):
        # TODO add label ID to name lookup
        return self._marques_labels

    @property
    def experiment_ids(self):
        # TODO add label ID to name lookup
        return self._experiment_ids

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            raise NotImplementedError("Fake data not implemented yet!")
            fake_image = [1] * 784
            fake_label = 0

            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._bouts = self.bouts[perm0]
            self._marques_labels = self.marques_labels[perm0]
            self._experiment_ids = self.experiment_ids
            # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            bouts_rest_part = self._bouts[start:self._num_examples]
            marques_labels_rest_part = self._marques_labels[start:self._num_examples]
            experiment_ids_rest_part = self._experiment_ids[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._bouts = self.bouts[perm]
                self._marques_labels = self.marques_labels[perm]
                self._experiment_ids = self.experiment_ids[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            bouts_new_part = self._bouts[start:end]
            marques_labels_new_part = self._marques_labels[start:end]
            experiment_ids_new_part = self._experiment_ids[start:end]
            bouts_batch = np.concatenate((bouts_rest_part, bouts_new_part), axis=0)
            marques_labels_batch = np.concatenate((marques_labels_rest_part, marques_labels_new_part), axis=0)
            experiment_ids_batch = np.concatenate((experiment_ids_rest_part, experiment_ids_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            bouts_batch = self._bouts[start:end]
            marques_labels_batch = self._marques_labels[start:end]
            experiment_ids_batch = self._experiment_ids[start:end]

        return bouts_batch, marques_labels_batch, experiment_ids_batch


def read_data_sets(data_archive,
                   # TODO not possible anymore?
                   experiments='all',
                   mmap=False,
                   padding=True,
                   max_bout_len=300,
                   fake_data=False,
                   dtype=dtypes.int16,
                   reshape=True,
                   test_fraction=0.1,
                   validation_fraction=0.05,
                   seed=None):
    '''
    Load bout data from `.npz` archive into Datasets object.

    Partitions the data into test, train and validation data. Expects bout data
    in 16 bit integer format and bout start indice data in 32bit integer
    format (created by the fishpy package).

    Parameters
    ----------
    data_archive : str
        Path to `npz` archive.
    experiments : {"all", list(experiment_names)}
        List of experiment names in `data_archive` to load for the data set. If
        "all", all experiments from `data_archive` are loaded.
    mmap : bool, optional
        If True, memory-map the `data_archive` file in readonly mode. A
        memory-mapped array is kept on disk and not loaded into memory.
        However, it can be accessed and sliced like any ndarray.
    padding : bool, optional
        If True, pad each bout with 0 in the frameIdx axis until all bouts have
        the same dimensions. If `max_bout_len` is "max_experiments", all
        bouts will have the dimensions of the longest bout in the specified
        experiment subset (`experiments` argument). If `max_bout_len` is
        "max_data", all bouts will have the dimensions of the longest
        bout in the entire `data_archive`.
    max_bout_len : int or str
        If int, drop all bouts with more frames then `max_bout_len`. If
        one of {"max_experiments", "max_data"} and `padding` is True, this
        specifies how much the single bouts are padded.
    fake_data : bool, optional
        If True, return fake data. For testing.
    dtype : {np.int16, np.float32}
        Datatype for bout data. If int16 is chosen, the tail fragments are left
        as [-32767, 32767] integers. If float32 is chosen, data is transferred
        into radian.
    reshape : bool, optional
        If True, flatten bout data per bout (shape=(num_bouts, bout_len *
        num_fragments)). If False (default), leave as shape (num_bouts,
        num_tail_fragments, num_frames, 1).
    test_fraction : float, optional
        Fraction of data to use as test set.
    validation_fraction : float, optional
        Fraction of data to use as validation set.
    seed : int, optional
        Set the seed for partitioning the data into train, test and validation
        set (for reproducibility). If None (default), a random seed is chosen
        and the partitioning is different for each run.

    Returns
    -------
    Datasets
        Collection of DataSet objects for train, test and validation data.
    '''
    if fake_data:

        print("Loading fake data for testing.")
        def fake():
            return DataSet([], [], fake_data=True, dtype=dtype)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    if not os.path.splitext(data_archive)[1] == '.npz':
        raise ValueError('Expected .npz file for data_archive, got {}'
                         .format(data_archive))

    if not 0 <= validation_fraction + test_fraction <= 1:
        raise ValueError(
            'Sum of validation and test fraction should be between 0 and 1. '
            'Received: {} and {}.'.format(validation_fraction, test_fraction))
    
    assert isinstance(mmap, bool)
    if mmap:
        mmap_mode = 'r'
    else:
        mmap_mode = None

    print("Loading dataset ...")
    start = time.time()
    # NpZFile object (data is not loaded to memory yet)
    with np.load(data_archive, mmap_mode=mmap_mode) as data:
        bouts = data['bouts']
        marques_labels = data['marques_label']
        experiment_ids = data['experiment_id']
        experiment_id_name_lookup = data['experiment_id_name_lookup'].item()
        marques_label_name_lookup = data['marques_label_name_lookup'].item()
        int16_to_rad_factor = data['int16_to_rad_factor']
    print("... took {} s".format(time.time() - start))

    num_bouts = bouts.shape[0]

    # TODO create data with split train and test part and avoid shuffling
    print("Shuffeling the data ...")
    start = time.time()
    np.random.seed(seed)
    perm = np.arange(num_bouts)
    np.random.shuffle(perm)
    bouts = bouts[perm]
    marques_labels = marques_labels[perm]
    experiment_ids = experiment_ids[perm]
    print("... took {} s".format(time.time() - start))

    test_size = int(num_bouts * test_fraction)
    validation_size = int(num_bouts * validation_fraction)
    train_size = num_bouts - (validation_size + test_size)

    print('Splitting bout data into {} train ({}%), {} test ({}%) and {} ({}%) '
          'validation samples'.format(train_size,
                                      100 - (test_fraction + validation_fraction) * 100,
                                      test_size, test_fraction * 100,
                                      validation_size, validation_fraction * 100))

    test_bouts = bouts[:test_size]
    test_marques_labels = marques_labels[:test_size]
    test_experiment_ids = experiment_ids[:test_size]
    validation_bouts = bouts[test_size:test_size + validation_size]
    validation_marques_labels = marques_labels[test_size:test_size + validation_size]
    validation_experiment_ids = experiment_ids[test_size:test_size + validation_size]
    train_bouts = bouts[test_size + validation_size:]
    train_marques_labels = marques_labels[test_size + validation_size:]
    train_experiment_ids = experiment_ids[test_size + validation_size:]
    assert train_bouts.shape[0] == train_size

    train = DataSet(train_bouts, experiment_ids=train_experiment_ids,
                    marques_labels=train_marques_labels, dtype=dtype, reshape=reshape,
                    experiment_id_name_lookup=experiment_id_name_lookup,
                    marques_label_name_lookup=marques_label_name_lookup,
                    int16_to_rad_factor=int16_to_rad_factor)
    validation = DataSet(validation_bouts, experiment_ids=validation_experiment_ids,
                         marques_labels=validation_marques_labels, dtype=dtype,
                         reshape=reshape, experiment_id_name_lookup=experiment_id_name_lookup,
                         marques_label_name_lookup=marques_label_name_lookup,
                         int16_to_rad_factor=int16_to_rad_factor)
    test = DataSet(test_bouts, experiment_ids=test_experiment_ids,
                   marques_labels=test_marques_labels, dtype=dtype, reshape=reshape,
                   experiment_id_name_lookup=experiment_id_name_lookup,
                   marques_label_name_lookup=marques_label_name_lookup,
                   int16_to_rad_factor=int16_to_rad_factor)
    
    return base.Datasets(train=train, validation=validation, test=test)


def load_bouts(data_archive, reshape=False, **kwargs):
    return read_data_sets(data_archive, reshape=reshape, **kwargs)
load_bouts.__doc__ = read_data_sets.__doc__
# TODO signature is missing (for default values)


def get_bout(n, bout_data, label_type, dataset='train'):
    """Returns flat ndarray for random bout with label n of given label_type"""
    if label_type not in ['marques_classification', 'experiment_id']:
        raise ValueError('`label_type` has to be one of ["marques_classification", '
                         '"experiment_id"], got {}'.format(label_type))

    data = getattr(bout_data, dataset)
    SIZE = 500
    for _ in range(int(np.ceil(data.num_examples / SIZE))):
        bouts, marques_labels, experiment_ids = data.next_batch(SIZE)
        if label_type == 'marques_classification':
            labels = marques_labels
        elif label_type == 'experiment_id':
            labels = experiment_ids
        idxs = iter(random.sample(range(SIZE), SIZE)) # non-in-place shuffle

        for i in idxs:
            if labels[i] == n:
                return bouts[i] # first match

    raise ValueError('label {} of label type {} not found in {} dataset'
                     .format(n, label_type, dataset))
