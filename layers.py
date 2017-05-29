import tensorflow as tf
from utils import variable_summaries


class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity, summarize_params=True):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity
        self.summarize_params = summarize_params

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            if not hasattr(self, 'w'):  # initialize weights first time
                self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                self.w = tf.nn.dropout(self.w, self.dropout)
            activation = self.nonlinearity(tf.matmul(x, self.w) + self.b)
            if self.summarize_params:
                with tf.name_scope('weights'):
                    variable_summaries(self.w)
                with tf.name_scope('bias'):
                    variable_summaries(self.b)
        return activation

    @staticmethod
    def wbVars(fan_in: int, fan_out: int):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))
