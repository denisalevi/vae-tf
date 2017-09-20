import os
import glob
import numpy as np

import tensorflow as tf

from vae_tf.mnist_helpers import load_mnist
from vae_tf.vae import VAE

from tf_cnnvis import activation_visualization, deconv_visualization, deepdream_visualization

import argparse
parser = argparse.ArgumentParser(description='Add deepdream activations to tensorboard images')
parser.add_argument('--log_folder', type=str, default='./log',
                    help='Where the vae model log folders are stored.')
parser.add_argument('--ckpt', type=str, default=None,
                    help='Checkpoint from wich to relaod the mdoel.')
args = parser.parse_args()

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = load_mnist()

# specify log folder manually
#META_GRAPH = './log_CNN/170713_1213_vae_784_conv-20x5x5-2x2-S_conv-50x5x5-2x2-S_conv-70x5x5-2x2-S_conv-100x5x5-2x2-S_10/'

# load trained VAE from last checkpoint
if args.ckpt:
    last_ckpt_path = args.ckpt
else:
    # load most recently trained vae model (assuming it hasen't been first reloaded)
    log_folders = [path for path in glob.glob('./' + args.log_folder + '/*') if not 'reloaded' in path]
    log_folders.sort(key=os.path.getmtime)
    META_GRAPH = log_folders[-1]
    last_ckpt_path = os.path.abspath(tf.train.latest_checkpoint(META_GRAPH))

print('last ckpt', last_ckpt_path)
print('Loading trained vae model from {}'.format(last_ckpt_path))
v = VAE(meta_graph=last_ckpt_path)

meta_graph = '.'.join([last_ckpt_path, 'meta'])

img = mnist.test.images[:1].reshape([1, 28, 28, 1]) + np.random.normal(0, 0.1, 28*28).reshape([1,28,28,1])

deepdream_visualization(
    graph_or_path=meta_graph,
    value_feed_dict={v.x_in: mnist.test.images[:1].reshape([1,28,28,1])},
    layer='encoding/Conv_2/convolution',
    classes=range(95),#[1, 2, 3, 4, 5],
    path_logdir=os.path.join(v.log_dir, 'viz')
)
