from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from layers import Dense
import plot
from utils import composeAll, print_


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 log_dir="./log"):
        """(Re)build a symmetric VAE model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if not meta_graph: # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"

            self.log_dir = os.path.join(os.path.abspath(log_dir), "{}_vae_{}".format(
                self.datetime, "_".join(map(str, self.architecture))))

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        else: # restore saved model
            # assuming meta_graph is the checkpoint file located in the models self.log_dir
            log_dir = os.path.dirname(os.path.realpath(meta_graph))
            prefix, model_name = os.path.basename(log_dir).split("_vae_")
            prefix_split = prefix.split("_reloaded")
            # when reloaded, the new log_dir will be
            # {datetime}_reloaded_{X}_{Y}_vae_{architecture} where X gets
            # incremented everytime a new reload is performed (indicating the source log dir)
            # and Y gets incremented when the folder already existst (making it unique)
            if len(prefix_split) == 1:
                source_idx = -1
            else:
                source_idx = int(prefix_split[1].split("_")[1])
            model_datetime = prefix_split[0]
            source_idx += 1
            unique_idx = 0
            while True:
                self.log_dir = os.path.join(os.path.join(log_dir, ".."),
                                            "{}_reloaded_{}_{}_vae_{}".format(
                                                model_datetime,
                                                source_idx,
                                                unique_idx,
                                                model_name)
                                           )
                if not os.path.isdir(self.log_dir):
                    break
                unique_idx += 1
            self.datetime = model_datetime
            model_architecture = re.split("_|-", model_name)
            self.architecture = [int(n) for n in model_architecture]

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(VAE.RESTORE_KEY)

        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_in, self.x_decoded,
         self.cost, self.global_step, self.train_op) = handles

        # Merge all the summaries and create writers
        self.merged_summaries = tf.summary.merge_all()
        print("Saving tensorBoard summaries in {}".format(self.log_dir))
        self.train_writer_dir = os.path.join(self.log_dir, 'train')
        self.validation_writer_dir = os.path.join(self.log_dir, 'validation')
        self.train_writer = tf.summary.FileWriter(self.train_writer_dir, self.sesh.graph)
        self.validation_writer = tf.summary.FileWriter(self.validation_writer_dir)

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.architecture[0]], name="x")
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        hidden_layers = self.architecture[1: -1]
        # encoding / "recognition": q(z|x)
        with tf.name_scope("encoding"):
            encoding = [Dense("dense{}".format(len(hidden_layers) - (n + 1)), hidden_size, dropout, self.nonlinearity)
                        # hidden layers reversed for function composition: outer -> inner
                        for n, hidden_size in enumerate(reversed(hidden_layers))]
            h_encoded = composeAll(encoding)(x_in)
        #h_encoded = tf.identity(h_encoded, name="x_encoded")

        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        with tf.name_scope("sample_latent"):
            z_mean = Dense("z_mean", self.architecture[-1], dropout)(h_encoded)
            z_log_sigma = Dense("z_log_sigma", self.architecture[-1], dropout)(h_encoded)

            # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
            z = self.sampleGaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        decoding = [Dense("dense{}".format(len(hidden_layers) - (n + 1)), hidden_size, dropout, self.nonlinearity)
                    for n, hidden_size in enumerate(hidden_layers)] # assumes symmetry
        # final reconstruction: restore original dims, squash outputs [0, 1]
        # prepend as outermost function
        decoding.insert(0, Dense("dense{}".format(len(hidden_layers)),
                                 self.architecture[0], dropout, self.squashing))

        with tf.name_scope("reconstructing"):
            x_reconstructed = composeAll(decoding)(z)
        #x_reconstructed = tf.identity(x_reconstructed, name="x_reconstructed")

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # reconstruction loss: mismatch b/w x & x_reconstructed
            # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
            rec_loss = tf.reduce_mean(VAE.crossEntropy(x_reconstructed, x_in))
            tf.summary.scalar('reconstruction_loss', rec_loss)

            # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
            kl_loss = tf.reduce_mean(VAE.kullbackLeibler(z_mean, z_log_sigma))
            tf.summary.scalar('KL_loss', kl_loss)

            # average over minibatch
            cost = tf.add(rec_loss, kl_loss)
            tf.summary.scalar('vae_cost', cost)
            cost += l2_reg
            tf.summary.scalar('regularized_cost', cost)

            # first add then reduce_mean
            #rec_loss = VAE.crossEntropy(x_reconstructed, x_in)
            #kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)
            #cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            #cost += l2_reg

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) # gradient clipping
                       for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            z_in = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]),
                                             shape=[None, self.architecture[-1]],
                                             name="latent_in")
        with tf.name_scope("decoding"):
            for dense_layer in decoding:
                # don't summarize params when decoding latent_in
                dense_layer.summarize_params = False
            x_decoded = composeAll(decoding)(z_in)
        #x_decoded = tf.identity(x_decoded, name="x_decoded")

        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed,
                z_in, x_decoded, cost, global_step, train_op)

    def sampleGaussian(self, mu, log_sigma):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

    def create_embedding(self, dataset, labels=None, label_names=None,
                         sample_latent=True, latent_space=True, input_space=True):
        """dataset.shape = (num_items, item_dimension)
        labels.shape = (num_items, num_labels)
        label_names = list"""

        if labels is not None:
            assert dataset.shape[0] == labels.shape[0]
        if not latent_space and not input_space:
            print("WARNING VAE.create_embedding called with input_space=False"
                  "and latent_space=False. No embedding created.")
            return

        # encode dataset
        mus, sigmas = self.encode(dataset)
        if sample_latent:
            emb = self.sesh.run(self.sampleGaussian(mus, sigmas))
        else:
            emb = mus

        self.embedding_vars = []
        if latent_space:
            emb_var_latent = tf.Variable(emb,
                                         name=('embedding_latent_sampled' if sample_latent
                                               else 'embedding_latent'),
                                         trainable=False)
            self.embedding_vars.append(emb_var_latent)
        if input_space:
            emb_var_input = tf.Variable(dataset, name='embedding_x_input', trainable=False)
            self.embedding_vars.append(emb_var_input)

        # since we create the embedding after training, we need to initialize the vars
        tf.variables_initializer(self.embedding_vars).run(session=self.sesh)

        # we need two configs for the train and validation log folders
        # (otherwise outmatic metadata loading won't work)
        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        configs = [projector.ProjectorConfig() for _ in range(2)]
        log_dirs = [self.train_writer_dir, self.validation_writer_dir]

        for config, log_dir in zip(configs, log_dirs):
            if latent_space:
                embedding_latent = config.embeddings.add()
                embedding_latent.tensor_name = emb_var_latent.name
            if input_space:
                embedding_input = config.embeddings.add()
                embedding_input.tensor_name = emb_var_input.name

            # create metadata file
            if labels is not None:
                header = ''
                if labels.ndim > 1:
                    if label_names is None:
                        label_names = ['label{}'.format(n) for n in range(labels.shape[1])]
                    err_msg = "label_names has to be of same length as there are columns in labels (got {} and {})"
                    assert len(label_names) == labels.shape[1], err_msg.format(len(label_names),
                                                                               labels.shape[1])
                    header = '\t'.join(label_names)
                metadata_file = os.path.join(log_dir, 'metadata.tsv')
                np.savetxt(metadata_file, labels, delimiter='/t', header=header)

                # Link this tensor to its metadata file (e.g. labels).
                if latent_space:
                    embedding_latent.metadata_path = metadata_file
                if input_space:
                    embedding_input.metadata_path = metadata_file

            # The next line writes a projector_config.pbtxt in the projector_dir. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)

        print("Created embeddings")

    @staticmethod
    def crossEntropy(obs, actual, offset=1e-7):
        # TODO: maybe use tf's cross entropy for stability?
        # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py#L111-L124
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)

    @staticmethod
    def l1_loss(obs, actual):
        """L1 loss (a.k.a. LAD), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l1_loss"):
            return tf.reduce_sum(tf.abs(obs - actual) , 1)

    @staticmethod
    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 -
                                        tf.exp(2 * log_sigma), 1)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sesh.run(zs) if is_tensor(zs) else zs) # coerce to np.array
            feed_dict.update({self.z_in: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_decoded, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate_every_n=None,
              verbose=True, save_final_state=True, plots_outdir="./png",
              plot_latent_over_time=False, plot_subsets_every_n=None, save_latent_embedding=True,
              save_input_embedding=True, save_summaries_every_n=None, **kwargs):

        if 'save' in kwargs.keys():
            raise TypeError("The `save` keyword was renamed to `save_final_state`!")
        elif kwargs:
            raise TypeError("train() got an unexpected keyword argument: {}".format(list(kwargs.keys())[0]))

        i_batch = 0
        avg_cost = None
        try:
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            if plot_latent_over_time: # plot latent space over log_BASE time
                BASE = 2
                INCREMENT = 0.5
                pow_ = 0

            while True:
                i_since_last_print = 0
                err_train = 0
                x, _ = X.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.dropout_: self.dropout}

                ### TRAINING
                if save_summaries_every_n is not None and i_batch % save_summaries_every_n == 0:
                    # save a summary checkpoint
                    fetches = [self.merged_summaries, self.x_reconstructed, self.cost,
                               self.global_step, self.train_op]
                    summary, x_reconstructed, cost, i_batch, _ = self.sesh.run(fetches, feed_dict)
                    self.train_writer.add_summary(summary, i_batch)
                else:  # no summary
                    fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                    x_reconstructed, cost, i_batch, _ = self.sesh.run(fetches, feed_dict)
                # TODO why calculate average cost? isn't current cost much more informative?
                err_train += cost
                i_since_last_print += 1
                avg_cost = err_train / i_since_last_print
                if verbose and i_batch % 1000 == 0:
                    # print average cost since last print
                    print("batch {} (epoch {}) --> cost (avg since last report): {}"
                          "".format(i_batch, X.train.epochs_completed, avg_cost))

                ### VALIDATION
                if cross_validate_every_n is not None and i_batch % cross_validate_every_n == 0:
                    # TODO change validation batch num / size / intervall if data less equal distributed
                    num_batches_validation = 1
                    validation_cost = 0
                    for n in range(num_batches_validation):
                        x, _ = X.validation.next_batch(self.batch_size)
                        feed_dict = {self.x_in: x}
                        fetches = [self.merged_summaries, self.x_reconstructed, self.cost]
                        summary, x_reconstructed, cost = self.sesh.run(fetches, feed_dict)
                        validation_cost += cost
                    validation_cost /= num_batches_validation
                    self.validation_writer.add_summary(summary, i_batch)
                    if verbose:
                        print("batch {} --> validation cost: {}".format(i_batch, validation_cost))
                    if plot_subsets_every_n is not None:
                        plot.plotSubset(self, x, x_reconstructed, n=10, name="cross_validation",
                                        outdir=plots_outdir)

                ### PLOTTING
                if plot_subsets_every_n is not None and i_batch % plot_subsets_every_n == 0:
                    # visualize `n` examples of current minibatch inputs + reconstructions
                    plot.plotSubset(self, x, x_reconstructed, n=10, name="train",
                                    outdir=plots_outdir)
                if plot_latent_over_time:
                    while int(round(BASE**pow_)) == i_batch:  # logarithmic time
                        plot.exploreLatent(self, nx=30, ny=30, ppf=True, outdir=plots_outdir,
                                           name="explore_ppf30_{}".format(pow_))

                        names = ("train", "validation", "test")
                        datasets = (X.train, X.validation, X.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self, dataset.images, dataset.labels, range_=
                                              (-6, 6), title=name, outdir=plots_outdir,
                                              name="{}_{}".format(name, pow_))

                        print("{}^{} = {}".format(BASE, pow_, i_batch))
                        pow_ += INCREMENT

                if i_batch >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("... training finished!")
                    break

        except KeyboardInterrupt:
            print("... training interrupted!")
            sys.exit(0)

        finally:
            print("final cost (@ step {} = epoch {}): {}".format(
                i_batch, X.train.epochs_completed, avg_cost))
            now = datetime.now().isoformat()[11:]

            self.create_embedding(X.train.images, labels=X.train.labels,
                                  latent_space=save_latent_embedding,
                                  input_space=save_input_embedding)

            if save_final_state:
                self.save_final_checkpoint()
            elif save_embedding:
                saver = tf.train.Saver(self.embedding_vars)
                outfile = os.path.join(self.log_dir, "embedding_checkpoint")
                print("Saving checkpoint for embeddings in {}".format(self.log_dir))
                saver.save(self.sesh, outfile, global_step=self.step)

            if save_summaries_every_n is not None:
                self.train_writer.flush()
                self.train_writer.close()
                self.validation_writer.flush()
                self.validation_writer.close()
            print("... done!")
            print("------- Training end: {} -------\n".format(now))

    def save_final_checkpoint(self):
        saver = tf.train.Saver(tf.global_variables())
        self.final_checkpoint = os.path.join(os.path.abspath(self.log_dir), "final_checkpoint")
        print("Saving Variables in {}".format(self.log_dir))
        outfile = saver.save(self.sesh, self.final_checkpoint, global_step=self.step)
        return outfile
