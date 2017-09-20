import os
import itertools
import fnmatch
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from scipy.cluster.hierarchy import dendrogram
from tensorflow.contrib.data.python.ops.dataset_ops import BatchDataset, PaddedBatchDataset

from vae_tf.utils import convert_into_grid, image_normalization

def plotSubset(model, x_in, x_reconstructed, n=10, cols=None, transform_data=None,
               tf_summary=None, save_png=None, show_plot=False, grid_dims=None,
               name="reconstruction", transform_kwargs=None):
    """Util to plot subset of inputs and reconstructed outputs"""
    # TODO wtf combing this with end-to-end reconstruction... or check what its used for in vae.py

    if tf_summary or save_png or show_plot:
        if tf_summary and not isinstance(tf_summary, str):
            # tf_summary == True, set default
            assert isinstance(tf_summary, bool), '`tf_summar` needs to be `str` or `bool`'
            tf_summary = model.validation_writer_dir

        if save_png and not isinstance(save_png, str):
            # save_png == True, set default
            assert isinstance(save_png, bool), '`save_png` needs to be `str` or `bool`'
            save_png = model.png_dir

        if transform_data is not None:
            assert callable(transform_data)
            images_in = []
            images_reconstructed = []
            for x in x_in:
                images_in.append(transform_data(x, **transform_kwargs))
            for x in x_reconstructed:
                images_reconstructed.append(transform_data(x, **transform_kwargs))
            images = np.vstack([images_in, images_reconstructed])
        else:
            images = np.vstack([x_in, x_reconstructed])

            print('images shape', images.shape)

        _create_grid_image(images, name, grid_dims=(2, n), tf_summary=tf_summary,
                           model=model, save_png=save_png, show_plot=show_plot)

def plot_reconstructions(model, data, n=10, transform_data=None, tf_summary=True,
                         save_png=None, show_plot=False, grid_dims=None,
                         datasets=['train', 'validation'], transform_kwargs=None):
    names = tuple(datasets)
    if isinstance(data, dict):
        datasets = [data[name] for name in datasets]
    else:  # assume its the old mnist DataSet class
        datasets = tuple([getattr(data, name) for name in names])
    with tf.variable_scope('reconstructions'):
        for name, dataset in zip(names, datasets):
            if isinstance(data, dict):
                if isinstance(dataset, (BatchDataset, PaddedBatchDataset)):
                    dataset = dataset.unbatch()
                dataset = dataset.shuffle(buffer_size=100 * n)
                dataset = dataset.batch(n)
                iterator = dataset.make_one_shot_iterator()
                x = model.sesh.run(iterator.get_next())
                if isinstance(x, tuple):
                    x = x[0]  # get the bout
            else:
                x, *_ = dataset.next_batch(n)
            x_reconstructed = model.vae(x)
            if tf_summary:
                if name == 'train':
                    tf_summary = model.train_writer_dir
                elif name == 'validation':
                    tf_summary = model.validation_writer_dir

            plotSubset(model, x, x_reconstructed, n=n, name=name,
                       tf_summary=tf_summary, save_png=save_png,
                       show_plot=show_plot, transform_data=transform_data,
                       transform_kwargs=transform_kwargs)

def plotInLatent(model, x_in, labels=[], range_=None, title=None,
                 save=True, name="data", outdir="."):
    """Util to plot points in 2-D latent space"""
    assert model.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
    title = (title if title else name)
    mus, _ = model.encode(x_in)
    ys, xs = mus.T

    plt.figure()
    plt.title("round {}: {} in latent space".format(model.step, title))
    kwargs = {'alpha': 0.8}

    classes = set(labels)
    if classes:
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs['c'] = [colormap[i] for i in labels]

        # make room for legend
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                    for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                    fancybox=True, loc='center left')

    plt.scatter(xs, ys, **kwargs)

    if range_:
        plt.xlim(*range_)
        plt.ylim(*range_)

    # plt.show()
    if save:
        model_name, *_ = model.get_new_layer_architecture(model.architecture)
        title = "{}_latent_{}_round_{}_{}.png".format(
            model.datetime, model_name, model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def exploreLatent(model, nx=20, ny=20, range_=(-4, 4), ppf=False,
                  save=True, name="explore", outdir="."):
    """Util to explore low-dimensional manifold of latent space"""
    assert model.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
    # linear range; else ppf (percent point function) == inverse CDF from [0, 1]
    range_ = ((0, 1) if ppf else range_)
    min_, max_ = range_
    dim = int(model.architecture[0]**0.5)

    # complex number steps act like np.linspace
    # row, col indices (i, j) correspond to graph coords (y, x)
    # rollaxis enables iteration over latent space 2-tuples
    zs = np.rollaxis(np.mgrid[max_:min_:ny*1j, min_:max_:nx*1j], 0, 3)

    if ppf: # sample from prior ~ N(0, 1)
        from scipy.stats import norm
        DELTA = 1E-16 # delta to avoid +/- inf at 0, 1 boundaries
        zs = np.array([norm.ppf(np.clip(z, DELTA, 1 - DELTA)) for z in zs])
    canvas = np.vstack([np.hstack([x.reshape([dim, dim]) for x in model.decode(z_row)])
                        for z_row in iter(zs)])

    plt.figure(figsize=(nx / 2, ny / 2))
    # `extent` sets axis labels corresponding to latent space coords
    plt.imshow(canvas, cmap="Greys", aspect="auto", extent=(range_ * 2))
    if ppf: # no axes
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis("off")
    plt.tight_layout()

    # plt.show()
    if save:
        model_name, *_ = model.get_new_layer_architecture(model.architecture)
        title = "{}_latent_{}_round_{}_{}.png".format(
            model.datetime, model_name, model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
        plt.close()


def interpolate(model, latent_1, latent_2, n=20, save=True, name="interpolate", outdir="."):
    """Util to interpolate between two points in n-dimensional latent space"""
    zs = np.array([np.linspace(start, end, n) # interpolate across every z dimension
                    for start, end in zip(latent_1, latent_2)]).T
    xs_reconstructed = model.decode(zs)

    dim = int(model.architecture[0]**0.5)
    canvas = np.hstack([x.reshape([dim, dim]) for x in xs_reconstructed])

    plt.figure(figsize = (n, 2))
    plt.imshow(canvas, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()

    # plt.show()
    if save:
        model_name, *_ = model.get_new_layer_architecture(model.architecture)
        title = "{}_latent_{}_round_{}_{}".format(
            model.datetime, model_name, model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
        plt.close()


def explore_latent_space_dimensions(model, amplitude, n=9, origin=None,
                                    name='explore_latent_dims', tf_summary=None,
                                    save_png=None, show_plot=False):
    """Vary only single dimensions in latent space"""
    
    latent_dims = model.architecture[-1]
    if origin is None:
        origin = np.zeros(latent_dims)
    all_xs =[]
    for i in range(latent_dims):
        latent_1 = origin.copy()
        latent_2 = origin.copy()
        latent_1[i] -= amplitude
        latent_2[i] += amplitude
        zs = np.array([np.linspace(start, end, n) # interpolate across every z dimension
                       for start, end in zip(latent_1, latent_2)]).T
        xs_reconstructed = model.decode(zs)
        all_xs.extend(xs_reconstructed)
    all_xs = np.stack(all_xs)

    if tf_summary or save_png or show_plot:
        grid_dims = (latent_dims, n)

        if tf_summary and not isinstance(tf_summary, str):
            # tf_summary == True, set default
            assert isinstance(tf_summary, bool), '`tf_summar` needs to be `str` or `bool`'
            tf_summary = model.validation_writer_dir

        if save_png and not isinstance(save_png, str):
            # save_png == True, set default
            assert isinstance(save_png, bool), '`save_png` needs to be `str` or `bool`'
            save_png = model.png_dir

        _create_grid_image(all_xs, name, grid_dims=grid_dims, tf_summary=tf_summary,
                           model=model, save_png=save_png, show_plot=show_plot)

def justMNIST(x, save=True, name="digit", outdir="."):
    """Plot individual pixel-wise MNIST digit vector x"""
    DIM = 28
    TICK_SPACING = 4

    fig, ax = plt.subplots(1,1)
    plt.imshow(x.reshape([DIM, DIM]), cmap="Greys",
               extent=((0, DIM) * 2), interpolation="none")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACING))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACING))

    # plt.show()
    if save:
        title = "mnist_{}.png".format(name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
        plt.close()


def morph(model, zs, n_per_morph=10, loop=True, save_png=None, show_plot=False,
          tf_summary=None, name="morph", grid_dims=None):
    '''
    Plot frames of morph between zs (np.array of 2+ latent points)

    Parameters
    ----------
    model
        VAE isntance
    zs
        list of latent space coords
    n_per_morph
    loop
    save_png
        If `None`, don't save. If `True`, save in `model.png_dir`, else
        `save_png` is assumed to be the directory to save in.
    show_plot
        If True, show matplotlib plot.
    tf_summary
        If None, don't create summary. If True, create tensorflow image summary
        in `model.validation_writer_dir`, else assume `tf_summary` is the
        target directory.
    save_every_number
        Save every morphed number as matplotlib png
    name
    grid_dims

    Returns
    -------
    '''
    assert len(zs) > 1, "Must specify at least two latent pts for morph!"

    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        # via https://docs.python.org/dev/library/itertools.html
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if loop:
        zs = np.append(zs, zs[:1], 0)

    all_xs = []
    num_rows = 0
    for z1, z2 in pairwise(zs):
        zs_morph = np.array([np.linspace(start, end, n_per_morph)
                             # interpolate across every z dimension
                             for start, end in zip(z1, z2)]).T
        xs_reconstructed = model.decode(zs_morph)
        all_xs.extend(xs_reconstructed)
        num_rows += 1
    #all_xs = np.stack(all_xs)

    if tf_summary or save_png or show_plot:
        if grid_dims == None:
            grid_dims = (num_rows, n_per_morph)
        else:
            assert grid_dims[0] * grid_dims[1] >= all_xs.shape[0],\
                    '`grid_dims` are to small to fit all images'

        if tf_summary and not isinstance(tf_summary, str):
            # tf_summary == True, set default
            assert isinstance(tf_summary, bool), '`tf_summar` needs to be `str` or `bool`'
            tf_summary = model.validation_writer_dir

        if save_png and not isinstance(save_png, str):
            # save_png == True, set default
            assert isinstance(save_png, bool), '`save_png` needs to be `str` or `bool`'
            save_png = model.png_dir

        _create_grid_image(all_xs, name, grid_dims=grid_dims, tf_summary=tf_summary,
                           model=model, save_png=save_png, show_plot=show_plot)

def _create_grid_image(images, name, grid_dims=None, tf_summary=None, model=None,
                       save_png=None, show_plot=False):
        assert images.ndim == 4,\
                '`images` needs to have 4 dims, got {}'.format(images.ndim)
        assert tf_summary is not None or save_png is not None or show_plot is not None,\
                'One of `tf_summary`, `save_png` and `show_plot` needs to be set'

        if grid_dims is None:
            # create square grid
            num_images = images.shape[0]
            dim = int(np.ceil(np.sqrt(num_images)))
            grid_dims = (dim, dim)

        # turn images into pixel range [0, 255]
        #images = image_normalization(images, ubound=255)
        # convert morphs into a single grid image
        grid = convert_into_grid(images, grid_dims=grid_dims)
        assert grid.ndim == 3
        grid = grid.reshape([1, *grid.shape])

        if tf_summary:
            assert isinstance(tf_summary, str), '`tf_summary` needs to be `str`'
            assert model is not None, 'For `tf_summary`, `model` argument is needed'

            image = tf.placeholder(tf.float32, shape=[None, None, None, None])
            image_summary = tf.summary.image(name, grid)
            summary_ran = model.sesh.run(image_summary, feed_dict={image : grid})

            file_writer = tf.summary.FileWriter(tf_summary)
            file_writer.add_summary(summary_ran)
            file_writer.close() # close file writer

        if save_png or show_plot:
            grid = 1 - grid  # tf.image.summary seems to invert the image
            plt.imshow(grid.reshape(*grid.shape[1:3]), cmap="Greys", interpolation='none')
            plt.gca().axis('off')

            if save_png:
                assert isinstance(save_png, str), '`save_png` needs to be `str`'

                try:
                    os.mkdir(save_png)
                except FileExistsError:
                    pass

                savefile = os.path.join(save_png, name + '.png')
                plt.savefig(savefile)
            
            if show_plot:
                plt.show()


# taken from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def plot_accuracies(accuracy_file, sort_by=None, index=None, kind='barh',
                    plot_columns=None, save_figure=None, show_figure=True,
                    left_adjust=None, annotate_bars=True, ticks=True,
                    **plot_kwargs):
    '''
    Visualize accuracies from a log file created by run_clustering.py

    Parameters
    ----------
    accuracy_file : str
        File created by run_clustering.py (with mean and std of multiple runs)
    sort_by : str, optional
        What to sort accuracies by. Can be 'arch' for architecture, 'beta' for the beta-VAE
        beta or any accuracy column name from the accuracy_file. Does not sort by default.
    index : dict, optional
        Only plot a specified index from the accuracy matrix. Keys can be
        column or index names, values can be single values or include UNIX
        wildcards ('*', '!', '?').
    kind : str, optional
        What kind of plot to use (default 'bar'), will be passed to pandas.DataFrame.plot().
    plot_columns : list(str), optional
        Which accuracy columns form the accuracy file to plot. By default plots all columns.
    save_figure : str, optional
        What to save the plot as (default None, doen't save).
    show_figure : bool, optional
        Weather or not to show the figure after creation.
    left_adjust : float, optional
        Subplots left margin (default None leaves matplotlib default).
    ticks : bool, optional
        If False, don't show y tick labels (default True).
    **plot_kwargs
        Keyword arguments past to pandas.DataFrame.plot().
    '''
    df = pd.read_csv(accuracy_file, delimiter='\t', index_col=[0, 1, 2, 3], comment='#')
    df_idx_slice = df
    if index is not None:
        for idx_level, idx in index.items():
            # idx needs to be str for `if '*' in idx` to not raise an error
            idx = str(idx)
            if '*' in idx or '?' in idx or '[!' in idx:
                if idx_level in df_idx_slice.index.names:
                    idx_in_idx_names = df_idx_slice.index.names.index(idx_level)
                    match_options = []
                    for df_idx in df_idx_slice.index:
                        idx_in_idx_names = df_idx_slice.index.names.index(idx_level)
                        match_options.append(df_idx[idx_in_idx_names])
                    idx_options = df_idx_slice.index
                elif idx_level in df_idx_slice.columns:
                    match_options = df_idx_slice[idx_level]
                    idx_options = match_options.index
                else:
                    raise ValueError('index key `{}` not in data'.format(idx_level))
                idx_matches = []
                for df_idx, option in zip(idx_options, match_options):
                    if fnmatch.fnmatch(str(option), idx):
                        idx_matches.append(df_idx)
                df_idx_slice = df_idx_slice.loc[idx_matches]
            else:
                if idx_level in df_idx_slice.index.names:
                    if idx_level == 'beta':
                        try:
                            idx = float(idx)
                        except ValueError:
                            raise ValueError("Got `{}` for `beta` index. Can't convert to float."
                                             .format(idx))
                    df_idx_slice = df_idx_slice.xs(idx, level=idx_level)
                elif idx_level in df_idx_slice.columns:
                    assert False, 'not implemented'
                else:
                    raise ValueError('index key `{}` not in data'.format(idx_level))

    # get DataFrame slices for mean and std
    df_mean_slice = df_idx_slice.xs('mean', level='stat')
    df_std_slice = df_idx_slice.xs('std', level='stat')

    mean = df_mean_slice
    std = df_std_slice
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]

        for sorter in reversed(sort_by):
            if sorter in df.index.names:
                mean = mean.sort_index(level=sorter, sort_remaining=False)#, ascending=False)
            elif sorter in df.columns:
                mean = mean.sort_values(by=sorter)#, ascending=False)
            else:
                raise ValueError("sort_by has to be 'arch', 'beta' or a column "
                                 "in accuracy_file, got {}".format(sort_by))
        # reindex std slice with indices from sorted mean slice
        std = df_std_slice.reindex(mean.index)

    # plot
    if plot_columns is None:
        plot_columns = ['clust_test_latent', 'clust_test_input', 'clust_train_latent']
    with plt.rc_context({'figure.autolayout': True}):
        axes = mean.plot(y=plot_columns,
                         xerr=std[plot_columns],
                         kind=kind,
                         **plot_kwargs)
        if not ticks:
            plt.tick_params(axis='y', labelleft='off')
        if annotate_bars:
            for p in axes.patches:
                axes.annotate('{:.3f}'.format(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
        axes.autoscale(tight=False)
        plt.tight_layout = True
        if left_adjust is not None:
            plt.gcf().subplots_adjust(left=left_adjust)
        if save_figure is not None:
            plt.savefig(save_figure)
        elif show_figure:
            plt.show()

def fig2rgb_array(fig, expand=False):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)

def plot_bout(bout, to_rad_transform=None, bout_len=None, cumsum=True,
              ylim=None, linewidth=1, alpha=0.5, savefig=None,
              figsize=(8, 2), cut_vertical=False, show_plot=False,
              remove_ticks=False):
    '''
    Create rgb array of plotted bout fragments over time for all fragments.

    Parameters
    ----------
    bout : ndarray
        A single bout of shape (num_fragments, num_frames), (num_fragments,
        num_frames, 1) or (1, num_fragments, num_frames, 1).
    to_rad_transform : tuple, optional
        If given, use to compute Pi in bouts representation and set plot ylim
        to (-Pi, Pi).
    bout_len : int, optional
        If given, set plot xlim to (0, bout_len).
    ylim : tuple, optional
        Plot y limit in multiples of Pi.

    Returns
    -------
    '''
    import seaborn as sns
    sns.set_style('white')
    #sns.set_context('paper')
    #sns.set_palette("hls", 8)
    if bout.ndim == 4:
        assert bout.shape[0] == bout.shape[3] == 1
        bout = bout.reshape(bout.shape[1:3])
    elif bout.ndim == 3:
        assert bout.shape[2] == 1
        bout = bout.reshape(bout.shape[0:2])
    else:
        assert bout.ndim == 2

    if bout.shape[0] <= 8:
        bout = bout.T
    assert bout.shape[1] <= 8

    if ylim is None:
        if cumsum:
            ylim = (-5, 5)
        else:
            ylim = (-1, 1)
    else:
        assert len(ylim) == 2

    #print('PLOT CALL before rescaling bout mean', bout.mean(), 'std', bout.std(), 'max', bout.max(), 'min', bout.min())
    if to_rad_transform is not None:
        scale, shift = to_rad_transform
        bout *= scale
        bout += shift
    #print('PLOT CALL after rescaling bout mean', bout.mean(), 'std', bout.std(), 'max', bout.max(), 'min', bout.min())

    # Pi in data representation
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if cumsum:
        bout = bout.cumsum(axis=1)[:, ::-1]
    ax.plot(bout, alpha=alpha, linewidth=linewidth)
    lw = linewidth / 2
    pi_magnitude = np.pi
    zero = 0
    #if to_rad_transform is not None:
    #    scale, shift = to_rad_transform
    #    pi_magnitude = np.pi / scale
    #    zero = - shift / scale
    ymin = zero - ylim[0] * pi_magnitude
    ymax = zero - ylim[1] * pi_magnitude
    #print('ymin', ymin, 'ymax', ymax)
    ax.set_ylim((ymin, ymax))
    ax.axhline(lw=lw, c='k', linestyle='--', y=zero - pi_magnitude)
    ax.axhline(lw=lw, c='k', linestyle='--', y=zero - 0.5 * pi_magnitude)
    ax.axhline(lw=lw, c='k', linestyle='--', y=zero)
    ax.axhline(lw=lw, c='k', linestyle='--', y=zero + 0.5 * pi_magnitude)
    ax.axhline(lw=lw, c='k', linestyle='--', y=zero + pi_magnitude)
    if bout_len is not None:
        ax.set_xlim(0, bout_len)
    max_x = ax.get_xlim()[1]
    for x in range(0, int(max_x), 50):
        ax.axvline(lw=lw, c='k', linestyle='--', x=x)
    if remove_ticks:
        ax.tick_params(
            axis='both',       # changes apply to the x-  and y-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',
            labelbottom='off',
            left='off',
            labelleft='off')
        # remove margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if show_plot:
        plt.show()

    if savefig is not None:
        plt.savefig(savefig)

    return fig

def plot_bout_as_rgb(bout, cut_vertical=False, show_plot=False, **plot_kwargs):
    '''
    Create rgb array of plotted bout fragments over time for all fragments.

    Parameters
    ----------
    bout : ndarray
        A single bout of shape (num_fragments, num_frames), (num_fragments,
        num_frames, 1) or (1, num_fragments, num_frames, 1).
    cut_vertical : bool, optional
        If True, cut the rgb array vertically and stack the result horizonally.
        If (width : height) ratio is (4 : 1), the resulting 2D array will be
        square.
    show_plot : bool, optional
        If True, show the created plot.
    **plot_kwargs
        Keyword arguments passed to `plot_bout` method.

    Returns
    -------
    ndarray
        3D array of shape (height, width, 3) with RGB values in the last
        dimension.
    '''
    if cut_vertical and to_rad_transform is None:
        print("WARNING: `cut_vertical` set without `to_rad_transform`. This might look shit.")

    fig = plot_bout(bout, show_plot=show_plot, **plot_kwargs)
    # to_rad_transform=None, bout_len=None, ylim=None, linewidth=3, alpha=0.5, figsize=(8, 2)

    rgb_array = fig2rgb_array(fig)
    plt.close(fig)

    if cut_vertical:
        width = rgb_array.shape[1]
        # Note: if width % 2 != 0, we loose the last pixel column (but who cares)
        half = int(width/2)
        first = rgb_array[:, :half]
        second = rgb_array[:, half:]
        # black vertical line
        idx = int(lw)
        if idx == 0:
            idx = 1
        first[-idx:] = np.zeros((idx, *first.shape[1:]))
        second[:idx] = np.zeros((idx, *second.shape[1:]))
        rgb_array = np.vstack([first, second])

    return rgb_array

def convert_bout_data_to_rgb_plots(bouts, bout_len, to_rad_transform,
                                   max_px, **kwargs):
    assert bouts.ndim == 4
    results = []
    resize_print = False
    print('convert bouts transform', to_rad_transform)
    print('converting plots into rgb arrays ...')
    for n, bout in enumerate(bouts):
        rgb = plot_bout_as_rgb(bout, cut_vertical=True, bout_len=bout_len,
                               to_rad_transform=to_rad_transform,
                               ylim=0.5 * np.array([-np.pi, np.pi]),
                               figsize=(8,2), alpha=0.5, linewidth=3,
                               remove_ticks=True, **kwargs)
        assert rgb.shape[0] == rgb.shape[1], 'rgb image not square'
        if n % 10 == 0:
            print('converted {}/{} plots'.format(n + 1, bouts.shape[0]))
        if rgb.shape[0] > max_px:
            rgb = cv2.resize(rgb, (max_px, max_px), interpolation=cv2.INTER_LINEAR)
            if not resize_print:
                print('resizing from {} to {} px'.format(rgb.shape[0], max_px))
                resize_print = True
        results.append(rgb)
    results = np.stack(results)
    return results
