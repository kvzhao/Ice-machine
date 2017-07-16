import os
import tensorflow as tf
import tensorflow.contrib as ct
import time

import numpy as np

from visualize import *
from utils import cal_energy, cal_defect_density
from scipy.misc import imsave

from constants import LEARNING_RATE_D
from constants import LEARNING_RATE_G
from constants import BETA1_D
from constants import BETA2_D
from constants import BETA1_G
from constants import BETA2_G
from constants import D_ITERS

from constants import NUM_BATCHES
from constants import BATCH_SIZE
from constants import EVAL_PER_ITERS
from constants import SAMPLE_PER_ITERS
from constants import SAVE_CKPT_PER_ITERS

from constants import TASK_NAME

class WassersteinGAN (object):
    def __init__ (self, G, D, x_sampler, z_sampler, scale=10.0):
        self.G = G
        self.D = D
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = D.x_dim
        self.z_dim = G.z_dim

        '''
            BUILD GRAPH
        '''

        self.x = tf.placeholder(tf.float32, [None,]+ self.x_dim, name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.G(self.z)
        # real
        self.d  = self.D(self.x, reuse=False)
        # fake
        self.d_ = self.D(self.x_)

        # compute WGAN loss
        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d ) - tf.reduce_mean(self.d_)

        eps = tf.random_uniform([], 0.0, 1.0)
        x_hat = eps * self.x + (1 - eps) * self.x_
        d_hat = self.D(x_hat)

        # compute gradient penalty
        ddx = tf.gradients(d_hat, x_hat)[0]
        #print(ddx.get_shape().as_list())
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        # loss function of formula (3)
        self.d_loss = self.d_loss + ddx

        # Solver
        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_D, beta1=BETA1_D, beta2=BETA2_D)\
                .minimize(self.d_loss, var_list=self.D.vars)
            
            self.g_adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_G, beta1=BETA1_G, beta2=BETA2_G)\
                .minimize(self.g_loss, var_list=self.G.vars)
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # logger file
        logfile = '/'.join(['logs', TASK_NAME])
        if tf.gfile.Exists(logfile):
            tf.gfile.DeleteRecursively(logfile)
        tf.gfile.MakeDirs(logfile)

        with tf.name_scope('summaries'):
            g_loss_sum = tf.summary.scalar('G_loss', self.g_loss)
            d_loss_sum = tf.summary.scalar('D_loss', self.d_loss)
            grad_penalty_sum = tf.summary.scalar('Grad Penalty', ddx)
            self.summary_op = tf.summary.merge_all()

        # Saver
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(logfile, self.sess.graph)



    def train(self):
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, NUM_BATCHES):
            
            for _ in range(D_ITERS):
                bx = self.x_sampler(BATCH_SIZE)
                bz = self.z_sampler(BATCH_SIZE, self.z_dim)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})
            
            bz = self.z_sampler(BATCH_SIZE, self.z_dim)
            self.sess.run(self.g_adam, feed_dict={self.z: bz, self.x: bx})

        
            if t % EVAL_PER_ITERS == 0:
                bx = self.x_sampler(BATCH_SIZE)
                bz = self.z_sampler(BATCH_SIZE, self.z_dim)

                '''
                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss, summary = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                '''
                d_loss, g_loss, summary = self.sess.run(
                    [self.d_loss, self.g_loss, self.summary_op], feed_dict={self.x: bx, self.z: bz}
                )

                bx_hat = self.sess.run(self.x_, feed_dict={self.z: bz})
                eng = cal_energy(bx_hat)
                dd = cal_defect_density(bx_hat)

                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                    (t, time.time() - start_time, d_loss, g_loss))

                print ('\t[ICE States]: Eval Total Energy %.6f, Defect Density %.6f' %  (eng, dd))

                self.writer.add_summary(summary, global_step=t)

            if t % SAMPLE_PER_ITERS == 0:
                bz = self.z_sampler(BATCH_SIZE, self.z_dim)
                ## save images
                bx = grid_transform(bx, self.x_dim)
                imsave('images/{}.png'.format(t/SAMPLE_PER_ITERS), bx)

            if t % SAVE_CKPT_PER_ITERS == 0:
                self.saver.save(self.sess, 'checkpoints/'+TASK_NAME, global_step=t)
    
    def sample(self):
        pass