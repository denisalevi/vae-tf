from datetime import datetime
import os
import re
import sys
import scipy
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import layers

from vae_tf import plot
from vae_tf.utils import print_, images_to_sprite, variable_summaries, get_deconv_params, convert_into_grid


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
        "squashing": tf.nn.sigmoid,
        # TODO add beta to file name
        "beta": 1.0,
        "img_dims": (28, 28),
        "weights_initializer": layers.xavier_initializer(),
        "biases_initializer": tf.zeros_initializer(),
        "reconstruction_loss": "crossEntropy",  # has to be a class method
        "data_range": (0, 1)
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 log_dir="./log", init=True, graph=None):
        """(Re)build a symmetric VAE model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000] (fully connected)

               To specify convolutinal layers, specify
               [num_filters, filter_shape, stride_shape, padding], where stride_shape and
               padding are optional (default to stride=(1,1), padding='SAME'). filter_shape
               and stride_shape must be tuple or int (assuming same in both dimensions).

               [1000, [32, (5,5), (1,1), 'SAME'], 10] specifies a VAE with 1000-D inputs,
               10-D latents space and a convolutional layer with 32 5x5 filters, stride (1, 1)
               and padding='SAME'. Using the default, [32, 5] would spcify the same layer. The
               deconvolution layer parameters are chosen such that the output dimensions fit
               the encoders input dimensions (possibly changing filter_shape and/or stride_shape).

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        # TODO use **hyperparams instead of d_hyperparams as __init__ arguments...
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)

        if not init:
            # for faster testing
            return

        self.sesh = tf.Session(graph=graph)

        if not meta_graph: # new model
            self.architecture = architecture

            model_name, layers, params = self.get_new_layer_architecture(architecture)
            self.hidden_layers = layers
            self.hidden_params = params

            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"

            self.log_dir = os.path.join(os.path.abspath(log_dir), "{}_vae_{}".format(
                self.datetime, model_name))

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
                self.log_dir = os.path.join(os.path.dirname(os.path.normpath(log_dir)),
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
            self.architecture, self.beta, img_dims = \
                    self.get_architecture_from_model_name(model_name)

            if img_dims is not None:
                self.img_dims = img_dims
            else:
                print("WARNING: reloaded model has depricated naming scheme. "
                      "Can't deduce image dimensions. Using {} (the value given "
                      "in the constructer if given, else the class default)"
                      .format(self.img_dims))

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(VAE.RESTORE_KEY)

        if self.img_dims[0] * self.img_dims[1] != self.architecture[0]:
            raise ValueError("img_dims attribute `{}` and `architecture[0]={}` don't fit. "
                             "The product of the former needs to be equal the latter."
                             .format(self.img_dims, self.architecture[0]))

        # unpack handles for tensor ops to feed or fetch
        self.unpack_handles(handles)

        # Merge all the summaries and create writers
        self.merged_summaries = tf.summary.merge_all()
        print("Saving tensorBoard summaries in {}".format(self.log_dir))
        self.train_writer_dir = os.path.join(self.log_dir, 'train')
        self.validation_writer_dir = os.path.join(self.log_dir, 'validation')
        self.train_writer = tf.summary.FileWriter(self.train_writer_dir, self.sesh.graph)
        self.validation_writer = tf.summary.FileWriter(self.validation_writer_dir)
        self.png_dir = os.path.join(self.log_dir, 'png')
        os.mkdir(self.png_dir)

    def get_new_layer_architecture(self, architecture):
        """Get the correct tf.contrib.layer method and parameters from the architecure list"""
        hidden_layers = []
        hidden_params = []
        layer_names = ['in-{}x{}'.format(*self.img_dims)]
        for n, layer in enumerate(architecture[1:-1]):
            if isinstance(layer, int):
                # fully connected layer
                hidden_layers.append('fully_connected')
                num_outputs = layer
                layer_names.append('fc-' + str(num_outputs))
                hidden_params.append({'num_outputs': num_outputs,
                                      'activation_fn': self.nonlinearity,
                                      'weights_initializer': self.weights_initializer,
                                      'biases_initializer': self.biases_initializer})
            elif isinstance(layer, (tuple, list)): # convolutional layer
                #err_msg = 'architecture[{}][0] must be list or tuple, is {}'.format(n+1, type(layer[0]))
                #assert isinstance(layer[0], tuple), err_msg
                # layer format: [num_filters, filter_shape, stride, padding]
                hidden_layers.append('convolution')

                # extract the convolutional parameters
                if len(layer) == 4:
                    num_filters, filter_shape, stride, padding = layer
                elif len(layer) == 3:
                    num_filters, filter_shape, tmp = layer
                    err_msg = 'architecture[{}][2] must be str, int or tuple(int), is {}'.format(n+1, type(tmp))
                    if isinstance(tmp, str):
                        padding = tmp
                        stride = (1, 1)  # default
                    elif isinstance(tmp, (tuple, int)):
                        stride = tmp
                        padding = 'SAME'  # default
                        if isinstance(stride, tuple) and not all([isinstance(s, int) for s in stride]):
                            raise ValueError(err_msg)
                    else:
                        raise ValueError(err_msg)
                elif len(layer) == 2:
                    num_filters, filter_shape = layer
                    stride = (1, 1)  # default
                    padding = 'SAME'  # default
                else:
                    raise ValueError('architecture[{}] (convolutional layer) must have lentgth 2, 3 or 4'
                                     ', has length {}'.format(n+1, len(layer)))

                err_msg = 'architecture[{}][{}] needs to be one of {}, but is {}'
                assert isinstance(num_filters, int), err_msg.format(n+1, 0, (int), type(num_filters))
                assert isinstance(filter_shape, (int, tuple)), err_msg.format(n+1, 1, (int, tuple), type(filter_shape))
                assert isinstance(stride, (int, tuple)), err_msg.format(n+1, 1, (int, tuple), type(stride))

                filter_shape = filter_shape if isinstance(filter_shape, tuple) else (filter_shape, filter_shape)
                stride = stride if isinstance(stride, tuple) else (stride, stride)

                # convolutional layer name format: ..._conv-NxF1xF2-S1xS2-P_...
                # where N - num_filters, F1/2 - filter_shape, S1/2 - stride, P - padding ('S' for 'SAME', 'V' for 'VALID')
                layer_names.append('conv-'
                                   # filter shape ('NxF1xF2')
                                   + 'x'.join([str(num_filters)] + list(map(str, filter_shape))) + '-'
                                   # stride ('S1xS2')
                                   + 'x'.join((list(map(str, stride)))) + '-'
                                   # padding ('S' or 'V')
                                   + ('S' if padding.lower() == 'same' else 'V'))

                hidden_params.append({'num_outputs': num_filters,
                                      'kernel_size': filter_shape,
                                      'stride': stride,
                                      'padding': padding,
                                      'activation_fn': self.nonlinearity,
                                      'weights_initializer': self.weights_initializer,
                                      'biases_initializer': self.biases_initializer})
            else:
                raise ValueError("architecure[{}]={} not understood. "
                                 "Has to be int (Dense) or tuple (Convolutional).".format(n+1, layer))

        layer_names.append('lat-{}'.format(architecture[-1]))
        model_name = '_'.join(layer_names)
        model_name = 'beta-{}_{}'.format(self.beta, model_name)
        return model_name, hidden_layers, hidden_params

    @staticmethod
    def get_architecture_from_model_name(model_name):
        """Extract architecture (as would be passed to self.__init__()) from model name string"""
        architecture = []
        beta = None
        img_dims = None
        layer_names = model_name.split("_")
        for name in layer_names:
            layer_type, *layer_params = name.split('-')
            if layer_type == 'beta':
                assert len(layer_params) == 1
                beta = float(layer_params[0])
            elif layer_type == 'in':
                assert len(layer_params) == 1
                img_h, img_w = layer_params[0].split('x')
                img_dims = tuple([int(img_h), int(img_w)])
                architecture.append(img_dims[0] * img_dims[1])
            elif layer_type == 'lat':
                assert len(layer_params) == 1
                latent_dims = int(layer_params[0])
                architecture.append(latent_dims)
            elif layer_type == 'fc':
                assert len(layer_params) == 1
                num_outputs = int(layer_params[0])
                architecture.append(num_outputs)
            elif layer_type == 'conv':
                assert len(layer_params) == 3
                filters, stride, padding = layer_params
                num_filters, *filter_shape = filters.split('x')
                num_filters = int(num_filters)
                filter_shape = tuple([int(f) for f in filter_shape])
                stride = tuple([int(s) for s in stride.split('x')])
                padding = 'SAME' if padding == 'S' else 'VALID'
                architecture.append([num_filters, filter_shape, stride, padding])
            elif len(layer_params) == 0:
                # old naming without fc- / conv- / in- / lat- but only int (fully connected)
                num_outputs = int(layer_type)
                architecture.append(num_outputs)
            else:
                raise ValueError("Couldn't split model name {} into architecture."
                                 "".format(model_name))
        return architecture, beta, img_dims


    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _build_encoder(self, x, dropout=1, verbose=True):
        # TODO why make a copy?
        encoder = tf.identity(x)
        if self.hidden_layers[0] == 'fully_connected':
            # for fully connected VAE we need to reshape the input
            encoder = tf.reshape(encoder, [-1, self.img_dims[0] * self.img_dims[1]])
        self.encoder_in_shapes = []
        for layer, params in zip(self.hidden_layers, self.hidden_params):
            # save encoder input shape as decoder output shape
            in_shape = encoder.get_shape().as_list()[1:]
            self.encoder_in_shapes.append(in_shape)
            if layer == 'fully_connected':
                if len(in_shape) == 4:
                    # we had a CONV before
                    raise NotImplementedError('Currently fully connected layers after '
                                              'convolutional layers are not supported!')
                assert len(in_shape) == 1
                encoder = layers.fully_connected(encoder, **params)
                encoder = layers.dropout(encoder, keep_prob=dropout)
                if verbose:
                    print("FC\n\tnum outputs {}\n\t output {}".format(
                            params["num_outputs"], encoder.get_shape().as_list()
                          ))
            elif layer == 'convolution':
                if len(in_shape) == 1:
                    # we had a FC before
                    print('in shape', in_shape)
                    raise NotImplementedError('Currently convolutional layers after fully '
                                              'connected layers are not supported!')
                assert len(in_shape) == 3
                encoder = layers.conv2d(encoder, **params)
                if verbose:
                    print("CONV\n\tnum filters {}\n\tfilter shape {}\n\tstride {}\n\tpadding {}"
                          "\n\tinput {}\n\toutput {}".format(
                              params["num_outputs"], params["kernel_size"], params["stride"],
                              params["padding"], in_shape, encoder.get_shape().as_list()[1:]
                          ))

        # also save intput shape for latent space (output of first decoder layer)
        self.encoder_in_shapes.append(encoder.get_shape().as_list()[1:])

        return layers.flatten(encoder)

    def _build_decoder(self, z, dropout=1, verbose=True):
        # TODO why make a copy?
        decoder = tf.identity(z)
        # first layer out from latent space (TODO could also be convolutional layer)
        decoder = layers.fully_connected(decoder,
                                         num_outputs=int(np.prod(self.encoder_in_shapes[-1])),
                                         activation_fn=self.nonlinearity,
                                         weights_initializer=self.weights_initializer,
                                         biases_initializer=self.biases_initializer)
        decoder = layers.dropout(decoder, keep_prob=dropout)

        # reshape to equal last encoder hidden layer [batches, height, width, channels]
        decoder = tf.reshape(decoder, [-1] + self.encoder_in_shapes[-1])

        for n, (layer, params, out_shape) in enumerate(zip(reversed(self.hidden_layers),
                                                       reversed(self.hidden_params),
                                                       reversed(self.encoder_in_shapes[:-1]))):
            decoder_params = params.copy()
            if n == len(self.hidden_layers) - 1:
                # last layer
                decoder_params["activation_fn"] = self.squashing

            if layer == 'fully_connected':
                assert len(out_shape) == 1
                # change num outputs to output shape of decoder (input shape of encoder)
                decoder_params["num_outputs"] = out_shape[0]
                decoder = layers.fully_connected(decoder, **decoder_params)
                decoder = layers.dropout(decoder, keep_prob=dropout)
                if verbose:
                    print("FC\n\tnum outputs {}\n\t output {}".format(
                              out_shape[0], decoder.get_shape().as_list()
                          ))
            elif layer == 'convolution':
                assert len(out_shape) == 3
                # change num filters of decoder to number of channels of encoder
                decoder_params["num_outputs"] = out_shape[-1]
                # calculate filter, stride and padding to get desired output shape
                in_shape = decoder.get_shape().as_list()[1:3]
                filter_shape, stride_shape, padding = get_deconv_params(
                    out_shape[:-1], in_shape, params["kernel_size"], params["stride"])
                decoder_params["kernel_size"] = filter_shape
                decoder_params["stride"] = stride_shape
                decoder_params["padding"] = padding
                decoder = layers.conv2d_transpose(decoder, **decoder_params)
                for a, b in zip(out_shape, decoder.get_shape().as_list()[1:]):
                    assert a == b, 'deconvolution output shape is wrong'
                if verbose:
                    print("DECONV\n\tnum filters {}\n\tfilter shape {}\n\tstride {}\n\tpadding {}"
                          "\n\tdesired output {}\n\tactual output {}".format(
                              out_shape[-1], filter_shape, stride_shape, padding, out_shape[:-1],
                              decoder.get_shape().as_list()[1:]
                          ))
            else:
                assert False, 'got something other then fc or conv string'

        if self.hidden_layers[0] == 'fully_connected':
            # for fully connected VAE we need to reshape the output
            decoder = tf.reshape(decoder, [-1, self.img_dims[0], self.img_dims[1], 1])
        return decoder

    def _buildGraph(self):

        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        with tf.variable_scope("input_data"):
            # handle for a Dataset (tensorflow 1.2+ Dataset API) object
            # to switch btwn train and validation datasets
            dataset_in = tf.placeholder(tf.string, shape=[], name='dataset_handle')

            # the Iterator class iterates over the Dataset
            iterator = tf.contrib.data.Iterator.from_string_handle(
                dataset_in, output_types=tf.float32, output_shapes=[None, *self.img_dims, 1]
            )
            x_in = iterator.get_next(name='x')
            if isinstance(x_in, tuple):
                x_in = x_in[0]
            x_in = tf.reshape(x_in, [-1, *self.img_dims, 1])

        # encoding / "recognition": q(z|x)
        with tf.variable_scope("encoding"):
            h_encoded = self._build_encoder(x_in, dropout=dropout)

        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
            with tf.variable_scope("z_mean"):
                z_mean = layers.fully_connected(
                    inputs=h_encoded,
                    #activation_fn=tf.nn.relu,
                    #activation_fn=tf.nn.sigmoid,#None,
                    activation_fn=None,
                    # TODO have self.latent_space_dim as variables instead
                    num_outputs=self.architecture[-1])
                z_mean = tf.nn.dropout(z_mean, dropout)
            with tf.variable_scope("z_log_sigma"):
                z_log_sigma = layers.fully_connected(
                    inputs=h_encoded,
                    #activation_fn=tf.nn.relu,
                    #activation_fn=tf.nn.sigmoid,#None,
                    activation_fn=None,
                    num_outputs=self.architecture[-1])
                z_log_sigma = tf.nn.dropout(z_log_sigma, dropout)

        with tf.name_scope("sample_latent"):
            # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
            z = self.sampleGaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        with tf.variable_scope("decoding"):
            x_reconstructed = self._build_decoder(z, dropout=dropout)

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # reconstruction loss: mismatch b/w x & x_reconstructed
            # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
            x_reconstructed_flat = layers.flatten(x_reconstructed)
            x_in_flat = layers.flatten(x_in)
            reconstruction_loss_function = getattr(self, self.reconstruction_loss)
            rec_loss = tf.reduce_mean(reconstruction_loss_function(x_reconstructed_flat, x_in_flat))
            tf.summary.scalar('reconstruction_loss', rec_loss)

            # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
            kl_loss = tf.reduce_mean(VAE.kullbackLeibler(z_mean, z_log_sigma))
            tf.summary.scalar('KL_loss', kl_loss)

            # average over minibatch
            cost = tf.add(rec_loss, self.beta * kl_loss)
            #cost = rec_loss
            #cost = kl_loss
            tf.summary.scalar('vae_cost', cost)
            cost += l2_reg
            tf.summary.scalar('regularized_cost', cost)

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) # gradient clipping
                       for grad, tvar in grads_and_vars if grad is not None]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            z_in = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]),
                                             shape=[None, self.architecture[-1]],
                                             name="latent_in")
        with tf.variable_scope("decoding", reuse=True):
            x_decoded = self._build_decoder(z_in, dropout=dropout, verbose=False)

        # create summaries of weights and biases
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if var.name.endswith(('weights:0', 'biases:0')):
                variable_summaries(var)

        return (dataset_in, x_in, dropout, z_mean, z_log_sigma, x_reconstructed,
                z_in, x_decoded, cost, global_step, train_op, learning_rate)

    def unpack_handles(self, handles):
        """Assignes the operations returned from _build_graph() to class attributes. These must include:
        (dataset_in, x_in, x_decoded, x_reconstructed, z_mean, z_log_sigma, z_in, global_step, dropout_, cost, train_op)
        """
        (self.dataset_in, self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_in, self.x_decoded,
         self.cost, self.global_step, self.train_op, self.learning_rate_) = handles

    def sampleGaussian(self, mu, log_sigma, seed=None):
        """(Differentiably!) draw sample from Gaussian with given shape,
        subject to random noise epsilon

        :param mu: Mean of Gaussian
        :param log_sigma: Log standard deviation of Gaussian
        :param seed: If set, makes noise repeatable across sessions (default=None).
        """
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon", seed=seed)
            return mu + epsilon * tf.exp(log_sigma)  # N(mu, I * sigma**2)

    def create_embedding(self, dataset, labels=None, label_names=None,
                         sample_latent=False, latent_space=True, input_space=True,
                         create_sprite=True, invert_sprite=True):
        """dataset.shape = (num_items, item_dimension)
        labels.shape = (num_items, num_labels)
        label_names = list
        create_sprite: bool or collable that takes 4D data and returns 4D pixle files
        """

        if labels is not None:
            assert dataset.shape[0] == labels.shape[0]

        if not latent_space and not input_space:
            print("WARNING VAE.create_embedding called with input_space=False"
                  "and latent_space=False. No embedding created.")
            return

        #assert dataset.ndim == 4
        dataset = dataset.reshape([-1, *self.img_dims, 1])

        # encode dataset
        mus, sigmas = self.encode(dataset)
        if sample_latent:
            emb = self.sesh.run(self.sampleGaussian(mus, sigmas))
        else:
            emb = mus

        embedding_vars = []
        if latent_space:
            emb_var_latent = tf.Variable(emb,
                                         name=('embedding_latent_sampled' if sample_latent
                                               else 'embedding_latent'),
                                         trainable=False)
            embedding_vars.append(emb_var_latent)
        if input_space:
            emb_var_input = tf.Variable(dataset.reshape([-1, self.img_dims[0] * self.img_dims[1]]),
                                        name='embedding_x_input', trainable=False)
            embedding_vars.append(emb_var_input)

        # since we create the embedding after training, we need to initialize the vars
        self.sesh.run(tf.variables_initializer(embedding_vars))


        # we need two configs for the train and validation log folders
        # (otherwise outmatic metadata loading won't work)
        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        log_dirs = [self.train_writer_dir, self.validation_writer_dir]
        configs = [projector.ProjectorConfig() for _ in range(len(log_dirs))]

        if create_sprite:
            # create sprite image
            if callable(create_sprite):
                dataset = create_sprite(dataset.reshape((-1, *self.img_dims, 1)))
                sprite_single_img_dims = dataset.shape[1:3]
            else:
                sprite_single_img_dims = self.img_dims 
            print('single sprite img dims', sprite_single_img_dims)
            # reshape images into (N, width, height)
            #embedding_images = dataset.reshape((-1, *self.img_dims))
            #embedding_images = np.tile(dataset, (1,1,1,3))
            #sprite_image = images_to_sprite(embedding_images, invert=invert_sprite)
            sprite_image = convert_into_grid(dataset, padding=0, normalize=True)#invert=invert_sprite)
            assert sprite_image.shape[0] <= 8192 and sprite_image.shape[1] <= 8192,\
                    'sprite image too large, got {} px'.format(sprite_images.shape[:2])
            if invert_sprite:
                sprite_image = 1 - sprite_image
            if sprite_image.shape[-1] == 1:
                # repeate last dimensions 3 times (gray scale) for imsave
                sprite_image = np.tile(sprite_image, 3)

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
                np.savetxt(metadata_file, labels, delimiter='\t', header=header, comments='')

                # Link this tensor to its metadata file (e.g. labels).
                if latent_space:
                    embedding_latent.metadata_path = metadata_file
                if input_space:
                    embedding_input.metadata_path = metadata_file

            if create_sprite:
                sprite_file = os.path.join(log_dir, 'sprite_img.png')
                scipy.misc.imsave(sprite_file, sprite_image)
                if latent_space:
                    embedding_latent.sprite.image_path = sprite_file
                    embedding_latent.sprite.single_image_dim.extend(list(sprite_single_img_dims))
                if input_space:
                    embedding_input.sprite.image_path = sprite_file
                    embedding_input.sprite.single_image_dim.extend(list(sprite_single_img_dims))

            # The next line writes a projector_config.pbtxt in the projector_dir. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)

        # we need to save a new checkpoint with the embedding variables
        self.save_checkpoint()
        print("Created embeddings")

    def scale_data_to_prob(self, data):
        '''
        Scale data to range [0, 1] to interpret values as probabilities.
        '''
        data_min, data_max = self.data_range
        if data.dtype.is_integer:
            if data.dtype.size < 2:
                # we have 8 or 16 bit ingegers
                data = data.as_float(data)
            elif data.dtype.size == 4:
                # we have 32 bit integer
                data = data.as_double(data)
            else:
                # we have 64 bit integer
                print("WARNING: casting 64 bit integer to 64 bit float. If your "
                      "data contains integers higher then 2^(53), they might "
                      "be not be representable by 64 bit floats and be rounded "
                      "to the next represantable number. (e.g. "
                      "``np.float64(np.int64(2**53 + 1)) == np.float64(np.int64(2**53))``)")
                data = data.as_double(data)
        # (data - min) / (max - min)
        prob = tf.div(tf.subtract(data, data_min), data_max - data_min)
        #tf.assert_positive(prob, message='data negative after scaling')
        #tf.assert_less_equal(prob, 1., message='data greater then 1 after scaling')
        return prob


    @staticmethod
    def crossEntropy(obs, actual, offset=1e-7):
        # TODO: maybe use tf's cross entropy for stability?
        # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py#L111-L124
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        epsilon = 1e-10
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_ + epsilon) +
                                  (1 - actual) * tf.log(1 - obs_ + epsilon), 1)

    @staticmethod
    def crossEntropy2(obs, actual, offset=1e-7):
        # TODO: maybe use tf's cross entropy for stability?
        # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py#L111-L124
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        epsilon = 1e-10
        # log distance reconstruction - decoder
        obs  = (obs  + 1) / 2
        actual = (actual + 1) / 2
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_ + epsilon) +
                                  (1 - actual) * tf.log(1 - obs_ + epsilon), 1)

    def lsd(self, a, b):
        '''Tensorflow log spectral-distance (works only on GPU)'''
        def tflog10(x):
            '''Tensorflow version of a log base 10'''
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator
        shape = tf.shape(a)
        #a = tf.reshape(a, (-1, *self.img_dims))
        #b = tf.reshape(b, (-1, *self.img_dims))
        psd = lambda x: tf.square(tf.cast(tf.abs(tf.fft(tf.cast(x, tf.complex64))), tf.float32))
        return tf.sqrt(tf.reduce_mean(tf.square(10*tflog10(psd(a)/psd(b)))))
    
    @staticmethod
    def sqrerr(actual, obs):
        '''Simple root mean square error'''
        with tf.name_scope("square_error"):
            loss_reco = tf.sqrt(tf.reduce_mean(tf.squared_difference(actual, obs)))
            return loss_reco

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

    def dataset_handle_from_tensors(self, x):
        '''Return string handle for one shot Iterator over Dataset created from Tensor x
        '''
        assert x.ndim == 4
        dataset = tf.contrib.data.Dataset.from_tensor_slices(x)
        # batch all tensors together
        dataset = dataset.batch(x.shape[0])
        # create Dataset handle to pass as feed_dict
        iterator = dataset.make_one_shot_iterator()
        handle = self.sesh.run(iterator.string_handle())
        return handle

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        if isinstance(x, tf.contrib.data.Dataset):
            iterator = x.make_one_shot_iterator()
            handle = self.sesh.run(iterator.string_handle())
        elif isinstance(x, np.ndarray):
            # create Dataset from given tensors
            # NOTE: if we run out of Graph memory here, use a placeholder for x
            #       and run iterator initializer with x as feed dict
            handle = self.dataset_handle_from_tensors(x)
        elif isinstance(x, bytes):
            handle = x
        elif isinstance(x, tf.Tensor) and x.dtype == tf.string:
            # assume its an iterator string handle that still has to be run
            handle = self.sesh.run(x)
        else:
            raise TypeError('`x` must be np.ndarray or tf.contrib.data.Dataset or a '
                            'string handle to it, got {}'.format(type(x).__name__))
        feed_dict = {self.dataset_in: handle}
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

    def train(self, train_dataset, max_iter=np.inf, max_epochs=None,
              cross_validate_every_n=None, validation_dataset=None, 
              verbose=True, save_final_state=True, plots_outdir=None,
              plot_latent_over_time=False, plot_subsets_every_n=None,
              save_summaries_every_n=None, shuffle=None, **kwargs):
        if 'save' in kwargs.keys():
            raise TypeError("The `save` keyword was renamed to `save_final_state`!")
        elif kwargs:
            raise TypeError("train() got an unexpected keyword argument: {}".format(list(kwargs.keys())[0]))

        if cross_validate_every_n is not None and validation_dataset is None:
            raise ValueError("Need `validation_dataset` for cross validation.")
        
        dataset_handles = {}
        iterators = {}
        for dataset, name in [(train_dataset, 'train'), (validation_dataset, 'validation')]:
            if name is 'train' or dataset is not None:
                if isinstance(dataset, np.ndarray):
                    dataset = tf.contrib.data.Dataset.from_tensor_slices(dataset)
                    dataset = dataset.batch(self.batch_size)
                elif not isinstance(dataset, tf.contrib.data.Dataset):
                    raise TypeError('`{}_dataset` needs to be a np.ndarray or '
                                    'tf.contrib.data.Dataset, got {}'
                                    .format(name, type(dataset).__name__))
                    
                # TODO can we check if dataset is batched and batch if not?

                if name == 'validation':
                    # we want to count epochs in train dataset (catch OutOfRangeError)
                    # for validation we can just repeat the epochs without signal
                    dataset = dataset.repeat()

                if shuffle is not None and name == 'test':
                    dataset = dataset.shuffle(shuffle)
                # TODO: if we wan't to shuffle at each epoch, add dataset.shuffle() 
                #       If added BEFORE dataset.repeat(), it will finish the data
                #       from one epoch before starting the next one.
                #       If added AFTER, if data from next epochs might come before
                #       first epoch is finished.
                # https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs

                iterator = dataset.make_initializable_iterator()
                iterators[name] = iterator

                handle, _ = self.sesh.run([iterator.string_handle(), iterator.initializer])
                dataset_handles[name] = handle

        if plots_outdir is None:
            plots_outdir = self.png_dir

        i_batch = self.step
        avg_cost = None
        try:
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            if plot_latent_over_time: # plot latent space over log_BASE time
                BASE = 2
                INCREMENT = 0.5
                pow_ = 0

            i_since_last_print = 0
            err_train = 0
            epochs_completed = 0
            start = time.time()
            print('Using learning rate:', self.learning_rate)
            while True:
                # try except to catch tf.errors.OutOfRangeError at end of epoch
                try:
                    feed_dict = {self.dataset_in: dataset_handles['train'],
                                 self.dropout_: self.dropout,
                                 self.learning_rate_: self.learning_rate}

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
                    if verbose and i_batch % 100 == 0:
                        # print average cost since last print
                        print("batch {} (epoch {}) --> cost (avg since last report): {} (took {:.2f}s)"
                              "".format(i_batch, epochs_completed, avg_cost, time.time() - start))
                        i_since_last_print = 0
                        err_train = 0
                        start = time.time()

                    ### VALIDATION
                    if cross_validate_every_n is not None and i_batch % cross_validate_every_n == 0:
                        # TODO change validation batch num / size / intervall if data less equal distributed
                        num_batches_validation = 1
                        validation_cost = 0
                        for n in range(num_batches_validation):
                            feed_dict = {self.dataset_in: dataset_handles['validation']}
                            if plot_subsets_every_n is None:
                                fetches = [self.merged_summaries, self.x_reconstructed, self.cost]
                                summary, x_reconstructed, cost = self.sesh.run(fetches, feed_dict)
                            else:
                                fetches = [self.merged_summaries, self.x_reconstructed, self.cost, self.x_in]
                                summary, x_reconstructed, cost, x_in  = self.sesh.run(fetches, feed_dict)
                            validation_cost += cost
                        validation_cost /= num_batches_validation
                        self.validation_writer.add_summary(summary, i_batch)
                        if verbose:
                            print("batch {} --> validation cost: {}".format(i_batch, validation_cost))
                        if plot_subsets_every_n is not None:
                            name = "reconstruction_cross_validation_step{}".format(self.step)
                            plot.plotSubset(self, x_in, x_reconstructed, n=10, name=name,
                                            save_png=plots_outdir)

                    ### PLOTTING
                    if plot_subsets_every_n is not None and i_batch % plot_subsets_every_n == 0:
                        # visualize `n` examples of current minibatch inputs + reconstructions
                        name = "reconstruction_train_step{}".format(self.step)
                        plot.plotSubset(self, x, x_reconstructed, n=10, name=name,
                                        save_png=plots_outdir)
                    if plot_latent_over_time:
                        while int(round(BASE**pow_)) == i_batch:  # logarithmic time
                            plot.exploreLatent(self, nx=30, ny=30, ppf=True, outdir=plots_outdir,
                                               name="explore_ppf30_{}".format(pow_))

                            names = ("train", "validation")
                            datasets = (train_dataset, validation_dataset)
                            for name, dataset in zip(names, datasets):
                                if datasets is not None:
                                    # TODO add labels argument to train method
                                    plot.plotInLatent(self, dataset, labels=[], range=(-6, 6),
                                                      title=name, outdir=plots_outdir,
                                                      name="{}_{}".format(name, pow_))

                            print("{}^{} = {}".format(BASE, pow_, i_batch))
                            pow_ += INCREMENT

                    if i_batch >= max_iter or epochs_completed >= max_epochs:
                        print("... training finished!")
                        break

                except tf.errors.OutOfRangeError:
                    epochs_completed += 1
                    # reinitialize train iterator
                    self.sesh.run(iterators['train'].initializer)

        except KeyboardInterrupt:
            print("\n... training interrupted!")

        except:
            print("\n... unxepected Error! Aborting.")
            raise

        finally:
            print("final cost (@ step {} = epoch {}): {}".format(
                i_batch, epochs_completed, avg_cost))
            now = datetime.now().isoformat()[11:]

            if save_final_state:
                self.save_checkpoint()

            if save_summaries_every_n is not None:
                self.train_writer.flush()
                self.train_writer.close()
                self.validation_writer.flush()
                self.validation_writer.close()
            print("------- Training end: {} -------\n".format(now))

    def save_checkpoint(self, name=None):
        checkpoint_name = 'checkpoint'
        if name is not None:
            checkpoint_name = name + '_' + 'checkpoint'
        saver = tf.train.Saver(tf.global_variables())
        final_checkpoint_name = os.path.join(os.path.abspath(self.log_dir), checkpoint_name)
        print("Saving checkpoint in {}".format(self.log_dir))
        self.final_checkpoint = saver.save(self.sesh, final_checkpoint_name, global_step=self.step)
