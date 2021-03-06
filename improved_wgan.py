import os
import tensorflow as tf
import tensorflow.contrib as ct
import time

import numpy as np

from visualize import *
from utils import *
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
        # E[D(G(z))]
        self.g_loss = tf.reduce_mean(self.d_)
        # E_x[D(x)] - E_z[D(G(z))] = E_x[D(x)] - E_z[D(x_)] 
        #self.d_loss = tf.reduce_mean(self.d ) - tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d_) - tf.reduce_mean(self.d)

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

        # logger file creation
        self.logfile = '/'.join(['logs', TASK_NAME])
        self.ckptfile = '/'.join(['checkpoints', TASK_NAME])

        if not tf.gfile.Exists(self.logfile):
            tf.gfile.MakeDirs(self.logfile)
            #tf.gfile.DeleteRecursively(self.logfile)
        if not tf.gfile.Exists(self.ckptfile):
            tf.gfile.MakeDirs(self.ckptfile)

        with tf.name_scope('summaries'):
            g_loss_sum = tf.summary.scalar('G_loss', self.g_loss)
            d_loss_sum = tf.summary.scalar('D_loss', self.d_loss)
            grad_penalty_sum = tf.summary.scalar('Grad Penalty', ddx)
            self.summary_op = tf.summary.merge_all()

        # Saver
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.logfile, self.sess.graph)

        # Checkpoints
        checkpoint = tf.train.get_checkpoint_state(self.ckptfile)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print ('Restore checkpoints from {}'.format(checkpoint.model_checkpoint_path))
        else:
            print ('Can not find checkpoint files')


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

                d_loss, g_loss, summary = self.sess.run(
                    [self.d_loss, self.g_loss, self.summary_op], feed_dict={self.x: bx, self.z: bz}
                )

                bx_ = self.sess.run(self.x_, feed_dict={self.z: bz})
                eng = cal_energy(bx_)
                dd = cal_defect_density(bx_)
                eng_th = cal_energy(binarization(bx_))
                dd_th  = cal_defect_density(binarization(bx_))
                # EVAL CORRELATION BETWEEN BATCHES

                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                    (t, time.time() - start_time, d_loss, g_loss))

                print ('[ICE States on Generated Data]: Eval Total Energy %.6f, Defect Density %.6f' %  (eng, dd))
                print ('[ICE States on Thresholed Data]: Eval Total Energy %.6f, Defect Density %.6f' %  (eng_th, dd_th))
                ## TODO: Save these results

                self.writer.add_summary(summary, global_step=t)

            if t % SAMPLE_PER_ITERS == 0:
                bz = self.z_sampler(BATCH_SIZE, self.z_dim)
                bx_ = self.sess.run(self.x_, feed_dict={self.z: bz})
                ## save images
                bx_ = grid_transform(bx_, self.x_dim)
                imsave('images/{}.png'.format(t/SAMPLE_PER_ITERS), bx_)

            if t % SAVE_CKPT_PER_ITERS == 0:
                self.saver.save(self.sess, self.ckptfile + '/model', global_step=t)
    
    def sample(self, num_of_batches=1, visualize=False):
        '''
            Sample generators and calculated mean energy and density
        '''
        engs=[]
        dds=[]
        engs_th=[]
        dds_th=[]
        for i in range(num_of_batches):
            bz = self.z_sampler(BATCH_SIZE, self.z_dim)
            bx_ = self.sess.run(self.x_, feed_dict={self.z: bz})
            np.save('/'.join(['tmp', 'bx_sampled_'+str(i)]), bx_)
            eng = cal_energy(bx_)
            dd = cal_defect_density(bx_)
            eng_th = cal_energy(binarization(bx_))
            dd_th  = cal_defect_density(binarization(bx_))
            engs.append(eng)
            engs_th.append(eng_th)
            dds.append(dd)
            dds_th.append(dd_th)
            print ('[%dth Batch Generated Data]: Eval Total Energy %.6f, Defect Density %.6f' %  (i, eng, dd))
            print ('[%dth Batch Thresholed Data]: Eval Total Energy %.6f, Defect Density %.6f' %  (i, eng_th, dd_th))
        return engs, engs_th, dds, dds_th