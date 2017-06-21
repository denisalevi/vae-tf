import glob
import os

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree, KDTree
import numpy as np

from vae import VAE
from clustering import hierarchical_clustering
from main import load_mnist

# suppress tf log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load most recently trained vae model (assuming it hasen't been first reloaded)
log_folders = [path for path in glob.glob('./log/*') if not 'reloaded' in path]
log_folders.sort(key=os.path.getmtime)
META_GRAPH = log_folders[-1]

# specify log folder manually
#META_GRAPH = './log/170620_2038_vae_784_500_500_10/'

# load mnist datasets
mnist = load_mnist()

# load trained VAE from last checkpoint (assuming only one to be there)
checkpoints = glob.glob(META_GRAPH + '/*.meta')
assert len(checkpoints) >> 0, 'no checkpoint file in log directory'
assert len(checkpoints) == 1, 'multiple checkpoint files in log directory'
last_ckpt_name = os.path.basename(checkpoints[0]).split('.')[0]
last_ckpt_path = os.path.abspath(os.path.join(META_GRAPH, last_ckpt_name))

print('Loading trained vae model from {}'.format(last_ckpt_path))
vae = VAE(meta_graph=last_ckpt_path)

## CLASSIFY BY CLUSTERING ENCODED TEST DATA (network trained on train data)
# encode mnist.test into latent space for clustering
# TODO any consequences of sampling here?
test_latent = vae.sesh.run(vae.sampleGaussian(*vae.encode(mnist.test.images)))

# do clustering
cluster_test = hierarchical_clustering(
    test_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
    plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
    num_clusters=10, max_dist=None, true_labels=mnist.test.labels
)

# calculate classification accuracy
accuracy = np.sum(mnist.test.labels == cluster_test.labels) / mnist.test.num_examples
print('Classification accuracy after hierachical clustering of test data is {}'.format(accuracy))

## CLASSIFY ENCODED TEST DATA BY USING CENTROIDS FROM CLUSTERING ENCODED TRAIN DATA
## (network trained on train data)
# encode mnist.train into latent space for clustering
# TODO any consequences of sampling here?
train_latent = vae.sesh.run(vae.sampleGaussian(*vae.encode(mnist.train.images)))

# do clustering
cluster_train= hierarchical_clustering(
    train_latent, linkage_method='ward', distance_metric='euclidean', check_cophonetic=False,
    plot_dir=os.path.join(vae.log_dir, 'cluster_plots'), truncate_dendrogram=50,
    num_clusters=10, max_dist=None, true_labels=mnist.train.labels
)

# classify test data by closest cluster centroid
kdtree = cKDTree(cluster_train.centroids)
dist, classify_test_labels = kdtree.query(test_latent, n_jobs=8)
#kdtree = KDTree(cluster_train.centroids)
#dist, classify_test_labels = kdtree.query(test_latent)

# calculate classification accuracy
accuracy2 = np.sum(mnist.test.labels == classify_test_labels) / mnist.test.num_examples
print('Classification accuracy after classification of test data with train data clusters is {}'.format(accuracy2))


# create embedding with cluster labels
labels = np.vstack([mnist.test.labels, cluster_test.labels, classify_test_labels]).T
vae.create_embedding(mnist.test.images, labels=labels,
                     label_names=['true_label', 'hierarchical_clustering', 'classification_from_train_clusters'],
                     sample_latent=True, latent_space=True, input_space=True,
                     image_dims=(28, 28))

