#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of DeepFM with tensorflow.

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import math
import time
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from utils.evaluation.RatingMetrics import *

__author__ = "Buracag Yang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Buracag Yang"
__email__ = "15591875898@163.com"
__status__ = "Development"


class DeepFM(object):
    def __init__(self, sess, num_user, num_item, **kwargs):
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.reg_rate = kwargs['reg_rate']
        self.num_factors = kwargs['num_factors']
        self.display_step = kwargs['display_step']
        self.show_time = kwargs['show_time']
        self.T = kwargs['T']
        self.layers = kwargs['layers']
        self.field_size = kwargs['field_size']

        self.train_features = None
        self.y = None
        self.dropout_keep = None
        self.first_oder_weight = None
        self.feature_embeddings = None
        self.feature_bias = None
        self.bias = None
        self.pred_rating = None
        self.pred = None
        self.loss = None
        self.optimizer = None
        self.num_training = None
        print("You are running DeepFM.")

    def build_network(self, feature_size):
        self.train_features = tf.placeholder(tf.int32, shape=[None, None])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.dropout_keep = tf.placeholder(tf.float32)
        self.first_oder_weight = tf.Variable(tf.random_normal([feature_size], mean=0.0, stddev=0.01))
        self.feature_embeddings = tf.Variable(tf.random_normal([feature_size, self.num_factors], mean=0.0, stddev=0.01))
        self.feature_bias = tf.Variable(tf.random_uniform([feature_size, 1], 0.0, 0.0))
        self.bias = tf.Variable(tf.constant(0.0))

        # f(x)
        with tf.variable_scope("First-order"):
            y1 = tf.reduce_sum(tf.nn.embedding_lookup(self.first_oder_weight, self.train_features), 1, keepdims=True)

        with tf.variable_scope("Second-order"):
            nonzero_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, self.train_features)
            sum_square = tf.square(tf.reduce_sum(nonzero_embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(nonzero_embeddings), 1)
            y_fm = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1, keepdims=True)
            y_fm = tf.nn.dropout(y_fm, self.dropout_keep)

        with tf.variable_scope("Deep_part"):
            deep_inputs = tf.reshape(nonzero_embeddings, shape=[-1, self.field_size*self.num_factors])  # None * (F*K)
            for i in range(len(self.layers)):
                deep_inputs = tf.contrib.layers.fully_connected(
                    inputs=deep_inputs, num_outputs=self.layers[i],
                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate), scope='mlp%d' % i)
                # TODO: dropout

            y_deep = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=1, activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
                scope='deep_out')
            y_d = tf.reshape(y_deep, shape=[-1, 1])

        with tf.variable_scope("DeepFM-out"):
            f_b = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_bias, self.train_features), 1)
            b = self.bias * tf.ones_like(self.y)
            self.pred_rating = tf.add_n([y1, y_fm, y_d, f_b, b])
            self.pred = tf.sigmoid(self.pred_rating)

        self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.pred_rating)) + \
            tf.contrib.layers.l2_regularizer(self.reg_rate)(self.feature_embeddings)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, train_data):
        self.num_training = len(train_data['Y'])
        total_batch = int(self.num_training / self.batch_size)
        rng_state = np.random.get_state()
        np.random.shuffle(train_data['Y'])
        np.random.set_state(rng_state)
        np.random.shuffle(train_data['X'])

        # train
        for i in range(total_batch):
            start_time = time.time()
            batch_y = train_data['Y'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_x = train_data['X'][i * self.batch_size:(i + 1) * self.batch_size]
            loss, _ = self.sess.run((self.loss, self.optimizer),
                                    feed_dict={self.train_features: batch_x,
                                               self.y: batch_y,
                                               self.dropout_keep: 0.5})
            if i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                if self.show_time:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        num_example = len(test_data['Y'])
        feed_dict = {self.train_features: test_data['X'], self.y: test_data['Y'], self.dropout_keep: 1.0}
        predictions = self.sess.run(self.pred_rating, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(test_data['Y'], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))
        _RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        print("RMSE:" + str(_RMSE))

    def execute(self, train_data, test_data):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            print("Epoch: %04d;" % epoch)
            self.train(train_data)
            if epoch % self.T == 0:
                self.test(test_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    # def predict(self, user_id, item_id):
    #     return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]
