import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracies(filename, sort_by=None, index=None, kind='bar',
                    plot_columns=None, save_figure=None,
                    rot=60, alpha=0.8, **plot_kwargs):
    """
    sort_by: {'arch', 'beta', 'clustering types'}, default None
    """
    df = pd.read_csv(filename, delimiter='\t', index_col=[0, 1, 2], comment='#')
    if index is not None:
        if sort_by == 'beta':
            idx_levels = ['arch']
        elif sort_by == 'arch':
            idx_levels = ['beta']
        else:
            idx_levels = ['arch', 'beta']

        error = False
        for level in idx_levels:
            try:
                # get the DataFrame slice for given index
                df_idx_slice = df.xs(index, level=level)
                error = False
                break
            except KeyError as err:
                error = err
                pass

        if error:
            raise KeyError("index has to be in {} if sort_by is {}"
                           ", got problem with: {} (maybe a type proble?)".format(
                               idx_levels, sort_by, error))

    else:  # index is None
        df_idx_slice = df

    # get DataFrame slices for mean and std
    df_mean_slice = df_idx_slice.xs('mean', level='stat')
    df_std_slice = df_idx_slice.xs('std', level='stat')

    if sort_by is not None:
        # sort the mean slice
        if sort_by in ['beta', 'arch']:
            mean = df_mean_slice.sort_index(level=sort_by, ascending=False)
        elif sort_by in df.columns:
            mean = df_mean_slice.sort_values(by=sort_by, ascending=False)
        else:
            raise ValueError("sort_by has to be 'arch', 'beta' or a column "
                             "in filename, got {}".format(sort_by))

        # reindex std slice with indices from sorted mean slice
        std = df_std_slice.reindex(mean.index)

    else:  # sort_by is None
        mean = df_mean_slice
        std = df_std_slice

    # plot
    if plot_columns is None:
        plot_columns = ['clust_test_latent', 'clust_test_input', 'clust_train_latent']
    with plt.rc_context({'figure.autolayout': True}):
        axes = mean.plot(y=plot_columns,
                         yerr=std[plot_columns],
                         kind=kind,
                         rot=rot,
                         alpha=alpha,
                         **plot_kwargs)
        axes.autoscale(tight=False)
        plt.tight_layout = True
        if save_figure is not None:
            plt.savefig(save_figure)
        else:
            plt.show()


if __name__ == '__main__':
    plot_accuracies('test_acc_file_created.txt')
    #plot_accuracies('test_acc_file_multi_idx.txt')
    #plot_accuracies('test_acc_file_multi_idx.txt', sort_by='arch')
    #plot_accuracies('test_acc_file_multi_idx.txt', sort_by='arch', index=0.9)
    #plot_accuracies('test_acc_file_multi_idx.txt', sort_by='beta', index='[500, 500, 8]')
    #plot_accuracies('test_acc_file_multi_idx.txt', kind='line')
    #plot_accuracies('test_acc_file_multi_idx.txt', sort_by='clustering_latent')
    #plot_accuracies('test_acc_file_multi_idx.txt', sort_by='clustering_input')
    #plot_accuracies('test_acc_file_multi_idx.txt', sort_by='clustering_input', index='[500, 500, 8]')
#    try:
#        plot_accuracies('test_acc_file_multi_idx.txt', sort_by='clustering_inpu')
#        raise ValueError('test failed')
#    except (KeyError, ValueError):
#        print('error excepted')
#        pass
#    try:
#        plot_accuracies('test_acc_file_multi_idx.txt', sort_by='beta', index=0.9)
#        raise ValueError('test failed')
#    except (KeyError, ValueError):
#        print('error excepted')
#        pass
#    try:
#        plot_accuracies('test_acc_file_multi_idx.txt', sort_by='arch', index='[500, 500, 8]')
#        raise ValueError('test failed')
#    except (KeyError, ValueError):
#        print('error excepted')
#        pass
#    try:
#        plot_accuracies('test_acc_file_multi_idx.txt', sort_by='clustering_input', index='[500, 500, 8, 2]')
#        raise ValueError('test failed')
#    except (KeyError, ValueError):
#        print('error excepted')
#        pass
#def plot_accuracies(filename, sort_by=None, index=None, kind='bar',
#                    plot_columns='all', rot=60, alpha=0.8, **plot_kwargs):
#    """
