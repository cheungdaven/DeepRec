#!/usr/bin/env python
"""Implementation of Deep Semantic Similarity Model with BPR.
Reference: Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Conference on information & knowledge management. ACM, 2013.
"""

import tensorflow as tf
import time
import numpy as np

from utils.evaluation.RankingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class DSSM():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=1024,
                 verbose=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("BPRMF.")

    def build_network(self, user_side_info, item_side_info, hidden_dim=100, output_size=30):

        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.neg_item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')
        self.y = tf.placeholder("float", [None], 'rating')

        self.user_side_info = tf.constant(user_side_info, dtype=tf.float32)
        self.item_side_info = tf.constant(item_side_info, dtype=tf.float32)

        user_input_dim = len(user_side_info[0])
        item_input_dim = len(item_side_info[0])

        user_input = tf.gather(self.user_side_info, self.user_id, axis=0)
        item_input = tf.gather(self.item_side_info, self.item_id, axis=0)
        neg_item_input = tf.gather(self.item_side_info, self.neg_item_id, axis=0)

        layer_1 = tf.layers.dense(inputs=user_input, units=user_input_dim,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dim, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dim, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_4 = tf.layers.dense(inputs=layer_3, units=hidden_dim, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        user_output = tf.layers.dense(inputs=layer_4, units=output_size, activation=None,
                                      bias_initializer=tf.random_normal_initializer,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        layer_1 = tf.layers.dense(inputs=item_input, units=item_input_dim,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dim, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dim, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_4 = tf.layers.dense(inputs=layer_3, units=hidden_dim, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        item_output = tf.layers.dense(inputs=layer_4, units=output_size, activation=None,
                                      bias_initializer=tf.random_normal_initializer,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.pred_rating = tf.reshape(output, [-1])

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        neg_item_latent_factor = tf.nn.embedding_lookup(self.Q, self.neg_item_id)

        self.pred_y = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1)
        self.pred_y_neg = tf.reduce_sum(tf.multiply(user_latent_factor, neg_item_latent_factor), 1)

        self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pred_y - self.pred_y_neg))) + self.reg_rate * (
        tf.norm(self.P) + tf.norm(self.Q))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        return self

    def prepare_data(self, train_data, test_data):
        '''
        You must prepare the data before train and test the model
        :param train_data:
        :param test_data:
        :return:
        '''
        t = train_data.tocoo()
        self.user = t.row.reshape(-1)
        self.item = t.col.reshape(-1)
        self.num_training = len(self.item)
        self.test_data = test_data
        self.total_batch = int(self.num_training / self.batch_size)
        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")
        return self

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        item_random_neg = []
        for u in user_random:
            neg_i = self.neg_items[u]
            s = np.random.randint(len(neg_i))
            item_random_neg.append(neg_i[s])

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item_neg = item_random_neg[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.neg_item_id: batch_item_neg})

            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        evaluate(self)

    def execute(self, train_data, test_data):

        self.prepare_data(train_data, test_data)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            self.train()
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch), end='')
                self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items
