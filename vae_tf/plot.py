import os
import itertools
import fnmatch

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram

def plotSubset(model, x_in, x_reconstructed, n=10, cols=None, outlines=True,
               save=True, name="subset", outdir="."):
    """Util to plot subset of inputs and reconstructed outputs"""
    n = min(n, x_in.shape[0])
    cols = (cols if cols else n)
    rows = 2 * int(np.ceil(n / cols)) # doubled b/c input & reconstruction

    plt.figure(figsize = (cols * 2, rows * 2))
    dim = int(model.architecture[0]**0.5) # assume square images

    def drawSubplot(x_, ax_):
        plt.imshow(x_.reshape([dim, dim]), cmap="Greys")
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for i, x in enumerate(x_in[:n], 1):
        # display original
        ax = plt.subplot(rows, cols, i) # rows, cols, subplot numbered from 1
        drawSubplot(x, ax)

    for i, x in enumerate(x_reconstructed[:n], 1):
        # display reconstruction
        ax = plt.subplot(rows, cols, i + cols * (rows / 2))
        drawSubplot(x, ax)

    # plt.show()
    if save:
        model_name, *_ = model.get_new_layer_architecture(model.architecture)
        title = "{}_batch_{}_round_{}_{}.png".format(
            model.datetime, model_name, model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
        plt.close()


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


def morph(model, zs, n_per_morph=10, loop=True, save=True, name="morph", outdir="."):
    """Plot frames of morph between zs (np.array of 2+ latent points)"""
    assert len(zs) > 1, "Must specify at least two latent pts for morph!"
    dim = int(model.architecture[0]**0.5) # assume square images

    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        # via https://docs.python.org/dev/library/itertools.html
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if loop:
        zs = np.append(zs, zs[:1], 0)

    all_xs = []
    for z1, z2 in pairwise(zs):
        zs_morph = np.array([np.linspace(start, end, n_per_morph)
                             # interpolate across every z dimension
                             for start, end in zip(z1, z2)]).T
        xs_reconstructed = model.decode(zs_morph)
        all_xs.extend(xs_reconstructed)

    for i, x in enumerate(all_xs):
        plt.figure(figsize = (5, 5))
        plt.imshow(x.reshape([dim, dim]), cmap="Greys")

        # axes off
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis("off")

        # plt.show()
        if save:
            model_name, *_ = model.get_new_layer_architecture(model.architecture)
            title = "{}_latent_{}_round_{}_{}.{}.png".format(
                model.datetime, model_name, model.step, name, i)
            plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
            plt.close()


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
