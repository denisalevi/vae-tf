from vae_tf.plot import plot_accuracies
import argparse

parser = argparse.ArgumentParser(description='Plot accuracies from a logfile created by run_clustering.py')
parser.add_argument('accuracy_file', nargs=1, type=str,
                    help='An accuracy file created by run_clustering.py')
parser.add_argument('--sort_by', nargs=1, default=None, type=str,
                    help='What to sort the accuracies by (beta, arch, some accuracy)')
parser.add_argument('--index', nargs=1, default=None, type=str,
                    help='Plot accuracies for a ceratin index in data matrix (one arch or beta)')
args = parser.parse_args()

plot_accuracies(args.accuracy_file[0], sort_by=args.sort_by[0], index=args.index[0])
