from numpy.testing import assert_raises
import numpy as np
from vae_tf.plot import plot_accuracies

def test_plot_accuracies():
    testdata = '''arch\tbeta\tstat\tnum_runs\tclust_test_latent\tclust_test_input\tclust_train_latent
[500, 500, 10]\t1.0\tmean\t3\t0.6258\t0.5899\t0.6948
[500, 500, 10]\t1.0\tstd\t3\t0.0232\t0.0274\t0.0000
[500, 500, 8]\t0.9\tmean\t3\t0.54\t0.32\t0.78
[500, 500, 8]\t0.9\tstd\t3\t0.032\t0.02\t0.12
[500, 500, 8]\t1.0\tmean\t3\t0.76\t0.56\t0.54
[500, 500, 8]\t1.0\tstd\t3\t0.023\t0.074\t0.0100'''

    testfile = '/tmp/testfile.txt'
    with open(testfile, 'w') as f:
        f.write(testdata)

    # just check if function calls pass
    plot_accuracies(testfile, show_figure=False)
    plot_accuracies(testfile, show_figure=False)
    plot_accuracies(testfile, show_figure=False, sort_by='arch')
    plot_accuracies(testfile, show_figure=False, sort_by='arch', index={'beta':0.9})
    plot_accuracies(testfile, show_figure=False, sort_by='beta', index={'arch':'[500, 500, 8]'})
    plot_accuracies(testfile, show_figure=False, kind='line')
    plot_accuracies(testfile, show_figure=False, sort_by='clust_test_latent')
    plot_accuracies(testfile, show_figure=False, sort_by='clust_test_input')
    plot_accuracies(testfile, show_figure=False, sort_by='clust_test_input', index={'arch':'[500, 500, 8]'})
    
    # check if wrong fucntion calls raise errors
    assert_raises(ValueError, plot_accuracies, testfile, sort_by='jibberish', show_figure=False)
    assert_raises(KeyError, plot_accuracies, testfile, sort_by='beta', index={'beta':0.9}, show_figure=False)
    assert_raises(ValueError, plot_accuracies, testfile, sort_by='arch',
                  index={'beta':'[500, 500, 8]'}, show_figure=False)
    assert_raises(ValueError, plot_accuracies, testfile, sort_by='clust_test_latent',
                  index={'beta':'[500, 500, 8, 2]'}, show_figure=False)

if __name__ == '__main__':
    test_plot_accuracies()
