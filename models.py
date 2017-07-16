import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import *

from constants import BATCH_SIZE
from constants import HEIGHT
from constants import WIDTH
from constants import CHANNEL
from constants import Z_DIM

class Discriminator(object):
    def __init__ (self):
        #self.x_dim = HEIGHT * WIDTH * CHANNEL
        self.x_dim = [HEIGHT, WIDTH, CHANNEL]
        self.name = 'models/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if (reuse):
                vs.reuse_variables()
            x = tf.reshape(x, [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL])
            conv1 = layers.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv2 = layers.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = layers.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = layers.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )

            conv4 = layers.flatten(conv4)
            ## why identity?
            fc = layers.fully_connected(conv4, 1, activation_fn=tf.identity)
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
    def __init__ (self):
        self.z_dim = Z_DIM
        self.x_dim = [HEIGHT, WIDTH, CHANNEL]
        self.name = 'models/g_net'
    
    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            fc = layers.fully_connected(z, 4 * 4 * 1024, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([BATCH_SIZE, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)
            conv2 = layers.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv3 = layers.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv4 = layers.conv2d_transpose(
                conv3, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv5 = layers.conv2d_transpose(
                conv4, CHANNEL, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh
            )
            return conv5


    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]