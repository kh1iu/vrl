import sys
sys.path.append('..')

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import os
import pickle
import ipdb
import math

from lib.saver import saver
from .conv import conv2D_32, upconv2D_32


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=0.5, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


class vrl_base(metaclass=ABCMeta):

    def __init__(self, sess, setup):

        self.sess = sess
        self.setup = setup        
        self.lr = setup['lr_init'] if 'lr_init' in setup else 1e-4
        self.lr_decay_step = setup['lr_decay_step'] if 'lr_decay_step' in setup else None
        if 'model_name' not in setup: setup['model_name'] = 'vrl'
        self.name = setup['model_name']
        self.image_patch_size = setup['image_patch_size']
        self.z_dim = setup['z_dim'] if 'z_dim' in setup else 2
        self.c1_patch_size = setup['c1_patch_size'] if 'c1_patch_size' in setup else self.image_patch_size[:-1]+[1]
        self.c2_patch_size = setup['c2_patch_size'] if 'c2_patch_size' in setup else self.image_patch_size[:-1]+[1]

        #Initialize VRL model
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            self._initialize()

        self.saver = saver(self.sess, self.setup)

        
    def _initialize(self):
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.image_pl = tf.placeholder(tf.float32,
                                       shape = [None]+self.image_patch_size,
                                       name='image_feed_pl')
        
        self.image_c1_pl = tf.placeholder(tf.float32,
                                          shape = [None]+self.c1_patch_size,
                                          name='image_c1_feed_pl')
        
        self.image_c2_pl = tf.placeholder(tf.float32,
                                          shape = [None]+self.c2_patch_size,
                                          name='image_c2_feed_pl')
        
        self.z_pl = tf.placeholder(tf.float32, shape = [None, self.z_dim], name='feature_feed_pl')

        self.z_mean, self.z_log_sigma_sq = self.post_fn(self.image_pl)
        
        self.eps_pl = tf.placeholder(tf.float32, shape = [None, self.z_dim], name='eps_feed_pl')

        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps_pl))

        self.image_rx = self.pred_fn(self.image_c1_pl, self.z)

        self.image_gen = self.pred_fn(self.image_c1_pl, self.z_pl)
            
        self.loss = self._build_loss()
        
        self.train_op = self._build_train_op(loss = self.loss,
                                             lr = self.lr,
                                             var_scope = self.name,
                                             global_step = self.global_step,
                                             lr_decay_step = self.lr_decay_step)

        theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.sess.run(tf.variables_initializer(theta))

        # Save tensor names
        self.setup['tensors'] = {
            'input_image_pl': self.image_pl.name,
            'input_image_c1_pl': self.image_c1_pl.name,
            'input_image_c2_pl': self.image_c2_pl.name,
            'z_pl': self.z_pl.name,
            'eps_pl': self.eps_pl.name,
            'image_rx_op': self.image_rx.name,
            'z_mean_op': self.z_mean.name,
            'z_log_sigma_sq_op': self.z_log_sigma_sq.name,
            'inference_op': self.z_mean.name,
            'image_gen_op': self.image_gen.name
        }

    def _build_train_op(self,
                       loss,
                       lr,
                       global_step=None,
                       opt=None,
                       var_scope=None,
                       lr_decay_step=None):

        with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE) as scope:

            if var_scope is None:
                theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                if type(var_scope) is not list:
                    var_scope = [var_scope]
                theta = []
                for v in var_scope:
                    theta += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=v)

            learning_rate = lr if lr_decay_step is None else tf.train.exponential_decay(lr, global_step, lr_decay_step , 0.5, staircase=True)

            if opt is None:
                opt = tf.train.AdamOptimizer(learning_rate)

            if var_scope is None:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            else:
                update_ops = []
                for v in var_scope:
                    update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=v)
                
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(loss = loss,
                                        global_step = global_step,
                                        var_list = theta)

        return train_op


    def train(self, imb, c1=None, c2=None):

        feed_dict = self._load_feed_dict(imb, c1, c2)
                    
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)

        return loss_value

    
    def validate(self, imb, c1=None, c2=None):

        feed_dict = self._load_feed_dict(imb, c1, c2)
        
        loss_value, imb_rx = self.sess.run([self.loss, self.image_rx], feed_dict = feed_dict)
        
        return loss_value, imb_rx

    def _load_feed_dict(self, imb, c1=None, c2=None):

        eps = np.random.randn(imb.shape[0], self.z_dim)

        imb_c1 = imb[..., :1] if c1 is None else c1
        imb_c2 = imb[..., 1:] if c2 is None else c2

        feed_dict = {self.image_pl: imb,
                     self.image_c1_pl: imb_c1,
                     self.image_c2_pl: imb_c2,
                     self.eps_pl: eps}        

        return feed_dict
    
    def num_updates(self):
        return tf.train.global_step(self.sess, self.global_step)

    @abstractmethod
    def _build_loss(self):
        pass

    @abstractmethod
    def post_fn(self, X, scope='post_params'):
        pass

    @abstractmethod
    def pred_fn(self, X, Z, scope='pred_params'):
        pass

        
class vrl2D_mlp_bce(vrl_base):

    def __init__(self, sess, setup):
        
        self._n_hidden = setup['n_hidden'] if 'n_hidden' in setup else 512
        super().__init__(sess, setup)

        
    def _initialize(self):
        
        super()._initialize()
        
        self.sigmoid_image_rx = tf.sigmoid(self.image_rx)
        self.sigmoid_image_gen = tf.sigmoid(self.image_gen)
        self.setup['tensors']['image_rx_op'] = self.sigmoid_image_rx.name
        self.setup['tensors']['image_gen_op'] = self.sigmoid_image_gen.name
        

    def _build_loss(self):

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.image_rx,
                                                            labels=self.image_c2_pl)
        
        rx_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        
        latent_loss = -0.5*tf.reduce_sum(1 + self.z_log_sigma_sq
                                         - tf.square(self.z_mean)
                                         - tf.exp(self.z_log_sigma_sq), 1)
                
        loss_fn = tf.reduce_mean(rx_loss + latent_loss)

        reg_loss_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_loss_list:
            loss_fn += tf.add_n(reg_loss_list)

        return loss_fn
    
    def validate(self, imb, c1=None, c2=None):
        
        feed_dict = self._load_feed_dict(imb, c1, c2)
        loss_value, imb_rx = self.sess.run([self.loss, self.sigmoid_image_rx], feed_dict = feed_dict)
        
        return loss_value, imb_rx

    def post_fn(self, X, scope='post_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:

            h = tf.layers.flatten(X)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=int(self._n_hidden/2),
                                activation=tf.nn.relu)
            
            z = tf.layers.dense(inputs = h,
                                units = 2*self.z_dim,
                                activation = None)

            z_mean = tf.slice(z, [0,0], [-1,self.z_dim])
            z_log_sigma_sq = tf.slice(z, [0,self.z_dim], [-1,self.z_dim])
            
            return z_mean, z_log_sigma_sq

        
    def pred_fn(self, X, Z, scope='pred_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            
            h = tf.concat([tf.layers.flatten(X), tf.layers.flatten(Z)], axis=1)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=int(self._n_hidden/2),
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)

            logits_x = tf.layers.dense(inputs=h,
                                       units=np.prod(self.c2_patch_size),
                                       activation=None)
            
            logits_x = tf.reshape(logits_x, [-1]+self.c2_patch_size)

            return logits_x
    

class vrl2D_mlp_mse(vrl_base):

    def __init__(self, sess, setup):
        self._n_hidden = setup['n_hidden'] if 'n_hidden' in setup else 512

        super().__init__(sess, setup)

    def _initialize(self):        
        super()._initialize()

    def _build_loss(self):

        latent_loss_reg = 1e-2

        rx_loss = tf.losses.mean_squared_error(self.image_c2_pl,
                                               self.image_rx)
        
        latent_loss = tf.reduce_mean(-0.5*tf.reduce_mean(1 + self.z_log_sigma_sq
                                                         - tf.square(self.z_mean)
                                                         - tf.exp(self.z_log_sigma_sq),1))
        
        loss_fn = tf.reduce_mean(rx_loss + latent_loss_reg*latent_loss)

        reg_loss_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_loss_list:
            loss_fn += tf.add_n(reg_loss_list)

        
        return loss_fn


    def post_fn(self, X, scope='post_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:

            h = tf.layers.flatten(X)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=int(self._n_hidden/2),
                                activation=tf.nn.relu)

            z_mean = tf.layers.dense(inputs=h,
                                     units=self.z_dim,
                                     activation=None)

            z_log_sigma_sq = 5 * tf.layers.dense(inputs=h,
                                                  units=self.z_dim,
                                                  activation=tf.tanh)

            return z_mean, z_log_sigma_sq
            
        
    def pred_fn(self, X, Z, scope='pred_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            
            h = tf.concat([tf.layers.flatten(X), tf.layers.flatten(Z)], axis=1)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=int(self._n_hidden/2),
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)

            x = tf.layers.dense(inputs=h,
                                units=np.prod(self.c2_patch_size),
                                activation=None)
            
            x = tf.reshape(x, [-1]+self.c2_patch_size)

            return x

        
class vrl2D_gumbel_bce(vrl_base):

    def __init__(self, sess, setup):

        self.sess = sess
        self.setup = setup
        
        self.lr = setup['lr_init'] if 'lr_init' in setup else 1e-4
        self.lr_decay_step = setup['lr_decay_step'] if 'lr_decay_step' in setup else None
        if 'model_name' not in setup: setup['model_name'] = 'vrl'
        self.name = setup['model_name']
        self.image_patch_size = setup['image_patch_size']
        self.z_dim = setup['z_dim']

        if 'N' not in setup:
            self.setup['N'] = 2
        if 'K' not in setup:
            self.setup['K'] = 5
            
        self.N = self.setup['N']
        self.K = self.setup['K']
        self.init_temp = 1.0
        self.min_temp = 0.5
        self.temp_anneal_rate = 5e-5 
        self._n_hidden = setup['n_hidden'] if 'n_hidden' in setup else 512
        self.c1_patch_size = setup['c1_patch_size'] if 'c1_patch_size' in setup else self.image_patch_size[:-1]+[1]
        self.c2_patch_size = setup['c2_patch_size'] if 'c2_patch_size' in setup else self.image_patch_size[:-1]+[1]

        #Initialize VRL model
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            self._initialize()

        self.saver = saver(self.sess, self.setup)

        
    def _initialize(self):
                
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.image_pl = tf.placeholder(tf.float32,
                                       shape = [None]+self.image_patch_size,
                                       name='image_feed_pl')
        
        self.image_c1_pl = tf.placeholder(tf.float32,
                                          shape = [None]+self.c1_patch_size,
                                          name='image_c1_feed_pl')
        
        self.image_c2_pl = tf.placeholder(tf.float32,
                                          shape = [None]+self.c2_patch_size,
                                          name='image_c2_feed_pl')

        # self.tau = tf.placeholder(tf.float32, [], name='temperature')

        self.tau = tf.compat.v1.placeholder_with_default(self.min_temp,
                                                         [],
                                                         name='temperature')
        
        self.z_logits = self.post_fn(self.image_pl)

        self.z_prob = tf.nn.softmax(tf.reshape(self.z_logits, [-1,self.K]))

        self.z = tf.reshape(
            gumbel_softmax(tf.reshape(self.z_logits, [-1,self.K]), self.tau),
            [-1, self.N*self.K])

        self.z_pl = tf.placeholder(tf.float32, shape = [None, self.N*self.K], name='feature_feed_pl')

        self.image_rx = self.pred_fn(self.image_c1_pl, self.z)

        self.image_gen = self.pred_fn(self.image_c1_pl, self.z_pl)

        self.sigmoid_image_rx = tf.sigmoid(self.image_rx)

        self.sigmoid_image_gen = tf.sigmoid(self.image_gen)
        
        self.loss = self._build_loss()
        
        self.train_op = super()._build_train_op(loss = self.loss,
                                                lr = self.lr,
                                                var_scope = self.name,
                                                global_step = self.global_step,
                                                lr_decay_step = self.lr_decay_step)

        theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.sess.run(tf.variables_initializer(theta))

        # Save tensor names
        self.setup['tensors'] = {
            'input_image_pl': self.image_pl.name,
            'input_image_c1_pl': self.image_c1_pl.name,
            'input_image_c2_pl': self.image_c2_pl.name,
            'z_pl': self.z_pl.name,
            'image_rx_op': self.sigmoid_image_rx.name,
            'image_gen_op': self.sigmoid_image_gen.name,
            'inference_op': self.z.name,
            'gumbel_temp_pl': self.tau.name
        }
                

    def _build_loss(self):

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.image_rx,
                                                            labels=self.image_c2_pl)
        
        rx_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

        kl_loss = tf.reduce_sum(
            tf.reshape(self.z_prob * tf.log(self.z_prob + 1e-20), [-1, self.N, self.K]),
            axis=[1,2])
                
        loss_fn = tf.reduce_mean(rx_loss + kl_loss)

        reg_loss_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_loss_list:
            loss_fn += tf.add_n(reg_loss_list)

        return loss_fn

    
    def train(self, imb, c1, c2):
            
        feed_dict = self._load_feed_dict(imb, c1, c2)

        _, loss_value = self.sess.run([self.train_op,
                                       self.loss],
                                      feed_dict = feed_dict)

        return loss_value

    
    def validate(self, imb, c1=None, c2=None):

        feed_dict = self._load_feed_dict(imb, c1, c2)
        
        loss_value, imb_rx = self.sess.run([self.loss, self.sigmoid_image_rx],
                                           feed_dict = feed_dict)
        
        return loss_value, imb_rx

    def _load_feed_dict(self, imb, c1=None, c2=None):

        _temp = max(self.init_temp*np.exp(-self.temp_anneal_rate*self.num_updates()), self.min_temp)

        imb_c1 = imb[..., :1] if c1 is None else c1
        imb_c2 = imb[..., 1:] if c2 is None else c2

        feed_dict = {self.image_pl: imb,
                     self.image_c1_pl: imb_c1,
                     self.image_c2_pl: imb_c2,
                     self.tau: _temp}

        return feed_dict

    
    def num_updates(self):
        return tf.train.global_step(self.sess, self.global_step)
    
    def post_fn(self, X, scope='post_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:

            h = tf.layers.flatten(X)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=int(self._n_hidden/2),
                                activation=tf.nn.relu)

            z = tf.layers.dense(inputs = h,
                                units = self.N*self.K,
                                activation = None)

            return z

        
    def pred_fn(self, X, Z, scope='pred_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            
            h = tf.concat([tf.layers.flatten(X), tf.layers.flatten(Z)], axis=1)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=int(self._n_hidden/2),
                                activation=tf.nn.relu)
            
            h = tf.layers.dense(inputs=h,
                                units=self._n_hidden,
                                activation=tf.nn.relu)

            logits_x = tf.layers.dense(inputs=h,
                                       units=np.prod(self.c2_patch_size),
                                       activation=None)
            
            logits_x = tf.reshape(logits_x, [-1]+self.c2_patch_size)

            return logits_x
        

class vrl2D_cnn_bce(vrl2D_mlp_bce):

    def __init__(self, sess, setup):
        self._n_filters = setup['n_hidden'] if 'n_hidden' in setup else 8
        super().__init__(sess, setup)


    def _initialize(self):
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.image_pl = tf.placeholder(tf.float32,
                                       shape = [None]+self.image_patch_size,
                                       name='image_feed_pl')
        
        self.image_c1_pl = tf.placeholder(tf.float32,
                                          shape = [None]+self.c1_patch_size,
                                          name='image_c1_feed_pl')
        
        self.image_c2_pl = tf.placeholder(tf.float32,
                                          shape = [None]+self.c2_patch_size,
                                          name='image_c2_feed_pl')
        
        self.z_pl = tf.placeholder(tf.float32, shape = [None, self.z_dim], name='feature_feed_pl')

        # self.bn_training = tf.placeholder(tf.bool,shape=(),name='bn_training')

        self.bn_training = tf.compat.v1.placeholder_with_default(False, [], name='bn_training')
        
        self.z_mean, self.z_log_sigma_sq = self.post_fn(self.image_pl,
                                                        bn_training=self.bn_training)
        
        self.eps_pl = tf.placeholder(tf.float32, shape = [None, self.z_dim], name='eps_feed_pl')

        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps_pl))

        self.image_rx = self.pred_fn(self.image_c1_pl,
                                     self.z,
                                     bn_training=self.bn_training)

        self.image_gen = self.pred_fn(self.image_c1_pl,
                                      self.z_pl,
                                      bn_training=False)

        self.sigmoid_image_rx = tf.sigmoid(self.image_rx)
        self.sigmoid_image_gen = tf.sigmoid(self.image_gen)
            
        self.loss = self._build_loss()
        
        self.train_op = self._build_train_op(loss = self.loss,
                                             lr = self.lr,
                                             var_scope = self.name,
                                             global_step = self.global_step,
                                             lr_decay_step = self.lr_decay_step)

        theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.sess.run(tf.variables_initializer(theta))

        # Save tensor names
        self.setup['tensors'] = {
            'input_image_pl': self.image_pl.name,
            'input_image_c1_pl': self.image_c1_pl.name,
            'input_image_c2_pl': self.image_c2_pl.name,
            'z_pl': self.z_pl.name,
            'eps_pl': self.eps_pl.name,
            'image_rx_op': self.sigmoid_image_rx.name,
            'z_mean_op': self.z_mean.name,
            'z_log_sigma_sq_op': self.z_log_sigma_sq.name,
            'inference_op': self.z_mean.name,
            'image_gen_op': self.sigmoid_image_gen.name,
            'bn_istraining_pl': self.bn_training.name,
        }

        
    def train(self, imb, c1=None, c2=None):

        feed_dict = self._load_feed_dict(imb, c1, c2)
        feed_dict[self.bn_training] = True
        _, loss_value = self.sess.run([self.train_op, self.loss],
                                      feed_dict = feed_dict)
        
        return loss_value

    def validate(self, imb, c1=None, c2=None):
        
        feed_dict = self._load_feed_dict(imb, c1, c2)
        feed_dict[self.bn_training] = False
        loss_value, imb_rx = self.sess.run([self.loss, self.sigmoid_image_rx], feed_dict = feed_dict)
        
        return loss_value, imb_rx


    def post_fn(self, X, bn_training, scope='post_params'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:

            h = conv2D_32(X, bn_training, filter_size=(3,3),
                          n_filters=self._n_filters, scope='conv')

            z = tf.layers.dense(inputs=tf.layers.flatten(h),
                                units=2*self.z_dim,            
                                activation=None)
            
            z_mean = tf.slice(z, [0,0], [-1,self.z_dim])
            z_log_sigma_sq = tf.slice(z, [0,self.z_dim], [-1,self.z_dim])
            
            return z_mean, z_log_sigma_sq

        
    def pred_fn(self, X, Z, bn_training, scope='pred_params'):
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:

            h = conv2D_32(X, bn_training, filter_size=(3,3),
                          n_filters=self._n_filters, scope='conv')
            
            xz = tf.layers.dense(inputs=tf.layers.flatten(h),
                            units=20,
                            activation=None)

            xz = tf.layers.batch_normalization(xz,
                                               training=bn_training)
            
            z = tf.concat([tf.layers.flatten(xz),
                           tf.layers.flatten(Z)], axis=1)

            z = tf.layers.dense(inputs=tf.layers.flatten(z),
                            units=np.prod(h.get_shape()[1:]),
                            activation=tf.nn.relu)

            z = tf.layers.batch_normalization(z,
                                              training=bn_training)

            z = tf.reshape(z, [-1] + list(h.get_shape()[1:]))

            y = upconv2D_32(z, bn_training, filter_size=(3,3),
                            n_filters=self._n_filters, scope='upconv')

            Y = tf.layers.conv2d(y,
                                 filters=1,
                                 kernel_size=1, 
                                 strides=1,
                                 activation=None)

            return Y
