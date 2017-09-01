#!/usr/bin/env python

from vae_tf.plot import plot_accuracies
import argparse

parser = argparse.ArgumentParser(description='Plot accuracies from a logfile created by run_clustering.py')
parser.add_argument('accuracy_file', nargs=1, type=str,
                    help='An accuracy file created by run_clustering.py')
parser.add_argument('--sort_by', nargs='+', default=None, type=str,
                    help='What to sort the accuracies by (beta, arch, some accuracy)')
parser.add_argument('--index', nargs='+', default=None, type=lambda kv: kv.split('='),
                    help=('Plot accuracies for indices in data matrix (form '
                          '--index key1=value1 key2=value2 ...), where key can '
                          'be column or index names and value can include UNIX '
                          'style wildcards (`*`, `!`, `?`)'))
parser.add_argument('--columns', nargs='+', default=None, type=str,
                    help='Which columns from the accuracy_file to plot (default is all)')
parser.add_argument('--kind', default='barh', type=str,
                    help='Which kind of plot to use. Default is "bar" for a barplot')
parser.add_argument('--left', default=None, type=float,
                    help='Ajust the subplot left margin.')
parser.add_argument('--no_numbers', action='store_false',
                    help='Dont annotate bars with values')
parser.add_argument('--no_ticks', action='store_false',
                    help='Dont annotate bars with values')
parser.add_argument('--kwargs', nargs='+', default=None, type=lambda kv: kv.split("="),
                    help=('keyword args past to pandas.DataFrame.plot() '
                          'function (form --kwargs key=value key2=value2)'))
args = parser.parse_args()

if args.index:
    index = {}
    for key, value in args.index:
        index[key] = value
else:
    index = None

plot_kwargs = {}
if args.kwargs:
    for key, value in args.kwargs:
        # if value is numeric, turn into float
        try:
            value = float(value)
        except ValueError:
            pass
        plot_kwargs[key] = value

plot_accuracies(args.accuracy_file[0], sort_by=args.sort_by, index=index,
                plot_columns=args.columns, kind=args.kind, left_adjust=args.left,
                annotate_bars=args.no_numbers, ticks=args.no_ticks, **plot_kwargs)
