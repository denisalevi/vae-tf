"""
Functions to standardize and rescale data

All functions have a keyword arguement `return_transforms`. If its True,
(scale, shift) is returned, such that the transformation can be repeated
by:

    f(x) = scale * x + shift

Data shapes are returned such that the broadcasting for equally shaped x
works correctly.
"""
import numpy as np

def standardize_per_fragment_dimension(bouts, return_transforms=False):
    """Standardization often used for audio signals
    Computes mean and std ignoring NaN values

    If `return_transforms` is True, returns (scale, shift), such that
    ``bouts_after_transform == bouts_before_transform * scale + shift``
    """
    shape = bouts.shape
    assert bouts.ndim == 4
    assert shape[1] == 8
    num_fragments = 8
    if bouts.dtype == np.int16:
        bouts = bouts.astype('float32')
    means = np.nanmean(bouts, axis=(0, 2, 3), keepdims=True)
    stds = np.nanstd(bouts, axis=(0, 2, 3), keepdims=True)
    # use inplace opreations to save memory
    bouts -= means
    bouts /= stds
    assert bouts.shape == shape
    if return_transforms:
        scales = 1. / stds
        shifts = - means / stds
        return scales, shifts

def standardize_per_data_point(bouts, return_transforms=False):
    """
    Computes mean and std ignoring NaN values

    If `return_transforms` is True, returns (scale, shift), such that
    ``bouts_after_transform == bouts_before_transform * scale + shift``
    """
    assert bouts.ndim == 4
    num_bouts = bouts.shape[0]
    if bouts.dtype == np.int16:
        bouts = bouts.astype('float32')
    # calculate mean and std per data point
    means = np.nanmean(bouts, axis=(1, 2, 3), keepdims=True)
    # we are calculating the variance of a random sample from the distribution
    # --> correct by using Bessel's correction (ddof = 1)
    stds = np.nanstd(bouts, axis=(1, 2, 3), ddof=1, keepdims=True)
    test = ((bouts[0] - means[0]) / stds[0])
    # use inplace operations to save memory
    bouts -= means
    bouts /= stds
    assert np.all(bouts[0] == test)
    if return_transforms:
        scales = 1. / stds
        shifts = - means / stds
        return scales, shifts

def standardize_by_entire_dataset(bouts, return_transforms=False):
    """
    Calculate mean and std over entire dataset.

    f(x) = (x - mean) / std
         = x / std - mean /std

    If `return_transforms` is True, returns (scale, shift), such that
    ``bouts_after_transform == bouts_before_transform * scale + shift``
    """
    assert bouts.ndim == 4
    mean = np.nanmean(bouts)
    std = np.nanstd(bouts)
    scale = 1. / std
    shift = - mean / std
    # use in-place operations to save memory
    bouts *= scale
    bouts += shift
    if return_transforms:
        return scale, shift

def rescale_data(bouts, low=0., high=1., return_transforms=False):
    """Rescales data to [low, high] in-place.

     f(x) = ((high - low) * (x - min)) / (max - min) + low
          = x * (high - low) / (max - min)
            - (high - low) / (max - min) * min + low

    If `return_transforms` is True, returns (scale, shift), such that
    ``bouts_after_transform == bouts_before_transform * scale + shift``
    """
    if bouts.dtype == np.int16:
        bouts = bouts.astype('float32')
    max_value = bouts.max()
    min_value = bouts.min()
    target_range = high - low
    current_range = max_value - min_value
    scale = target_range / current_range
    shift = low - target_range / current_range * min_value
    # use in-place operations to save memory
    bouts *= scale
    bouts += shift
    if return_transforms:
        return scale, shift

def calculate_reverse_transform(scales, shifts):
    """
    Calculate reverse transformation parameters from stacke linear transforms.

    For two linear transorms with scales = [a, c] and shifts = [b, d]
        Y = a * X + b
        Z = c * Y + d
    return parameter e and f such that
        X = e * Z + f
    """
    # describe all stacked linear transformations as one linear transformation
    # Y = a * X + b
    # Z = c * Y + d = a * c * X + b * c + d
    # new_scale = a * c
    # new_shift = b * c + d
    final_scale = 1
    final_shift = 0
    for scale, shift in zip(scales, shifts):
        # a : old_scale, b : old_shift, c : scale, d : shift
        final_scale = final_scale * scale  # new_scale = a * c
        final_shift = final_shift * scale + shift  # new_shift = b * c + d

    # reverse transform to get back to radian
    # Y = a * X + b  ->  X = (Y - b) / a = 1 / a * Y - b / a
    # rev_scale = 1 / a
    # rev_shift = - b / a
    rev_scale = 1 / final_scale
    rev_shift = - final_shift / final_scale

    return rev_scale, rev_shift

