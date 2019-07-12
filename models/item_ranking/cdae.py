#!/usr/bin/env python
"""Implementation of CDAE.
Reference: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings
of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.
"""

import tensorflow as tf
import time
import numpy as np

from utils.evaluation.RankingMetrics import evaluate

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class CDAE(object):
    def __init__(self, sess, num_user, num_item, learning_rate=0.01, reg_rate=0.01, epoch=500, batch_size=100,
                 verbose=False, t=1, display_step=1000):
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step

        self.user_id = None
        self.corrupted_rating_matrix = None
        self.rating_matrix = None
        self.corruption_level = None
        self.layer_2 = None
        self.loss = None
        self.optimizer = None
        self.train_data = None
        self.neg_items = None
        self.num_training = None
        self.total_batch = None
        self.test_data = None
        self.test_users = None
        self.reconstruction = None
        print("You are running CDAE.")

    def build_network(self, hidden_neuron=500, corruption_level=0):
        self.corrupted_rating_matrix = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item])
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item])
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None])
        self.corruption_level = corruption_level

        _W = tf.Variable(tf.random_normal([self.num_item, hidden_neuron], stddev=0.01))
        _W_prime = tf.Variable(tf.random_normal([hidden_neuron, self.num_item], stddev=0.01))
        _V = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        b = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b_prime = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        print(np.shape(tf.matmul(self.corrupted_rating_matrix, _W)))
        print(np.shape(tf.nn.embedding_lookup(_V, self.user_id)))
        layer_1 = tf.sigmoid(tf.matmul(self.corrupted_rating_matrix, _W) + tf.nn.embedding_lookup(_V, self.user_id) + b)
        self.layer_2 = tf.sigmoid(tf.matmul(layer_1, _W_prime) + b_prime)

        self.loss = - tf.reduce_sum(
            self.rating_matrix * tf.log(self.layer_2) + (1 - self.rating_matrix) * tf.log(1 - self.layer_2)) + \
            self.reg_rate * (tf.nn.l2_loss(_W) + tf.nn.l2_loss(_W_prime) + tf.nn.l2_loss(_V) +
                             tf.nn.l2_loss(b) + tf.nn.l2_loss(b_prime))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):
        self.train_data = self._data_process(train_data)
        self.neg_items = self._get_neg_items(train_data)
        self.num_training = self.num_user
        self.total_batch = int(self.num_training / self.batch_size)
        self.test_data = test_data
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(self.total_batch):
            start_time = time.time()
            if i == self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={
                self.corrupted_rating_matrix: self._get_corrupted_input(self.train_data[batch_set_idx, :],
                                                                        self.corruption_level),
                self.rating_matrix: self.train_data[batch_set_idx, :],
                self.user_id: batch_set_idx
                })
            if self.verbose and i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                if self.verbose:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.corrupted_rating_matrix: self.train_data,
                                                                     self.user_id: range(self.num_user)})

        evaluate(self)

    def execute(self, train_data, test_data):
        self.prepare_data(train_data, test_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            self.train()
            if epoch % self.T == 0:
                print("Epoch: %04d; " % epoch, end='')
                self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return np.array(self.reconstruction[np.array(user_id), np.array(item_id)])

    @staticmethod
    def _data_process(data):
        return np.asmatrix(data)

    def _get_neg_items(self, data):
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = [k for k, i in enumerate(data[u]) if data[u][k] == 0]
            # print(neg_items[u])

        return neg_items

    @staticmethod
    def _get_corrupted_input(input_train_data, corruption_level):
        return np.random.binomial(n=1, p=1 - corruption_level) * input_train_data


class ICDAE(object):
    """
    Based on CDAE and I-AutoRec, I designed the following item based CDAE, it seems to perform better than CDAE
    slightly.
    """
    def __init__(self, sess, num_user, num_item, learning_rate=0.01, reg_rate=0.01, epoch=500, batch_size=300,
                 verbose=False, t=2, display_step=1000):
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step

        self.corrupted_interact_matrix = None
        self.interact_matrix = None
        self.item_id = None
        self.corruption_level = None
        self.layer_2 = None
        self.loss = None
        self.optimizer = None
        self.train_data = None
        self.neg_items = None
        self.num_training = None
        self.total_batch = None
        self.test_data = None
        self.test_users = None
        self.reconstruction = None
        print("Item based CDAE.")

    def build_network(self, hidden_neuron=500, corruption_level=0):
        self.corrupted_interact_matrix = tf.placeholder(dtype=tf.float32, shape=[None, self.num_user])
        self.interact_matrix = tf.placeholder(dtype=tf.float32, shape=[None, self.num_user])
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None])
        self.corruption_level = corruption_level

        _W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))
        _W_prime = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        _V = tf.Variable(tf.random_normal([self.num_item, hidden_neuron], stddev=0.01))

        b = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b_prime = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        # print(np.shape(tf.matmul(self.corrupted_interact_matrix, _W)))
        # print(np.shape( tf.nn.embedding_lookup(_V, self.item_id)))
        layer_1 = tf.sigmoid(tf.matmul(self.corrupted_interact_matrix, _W) + b)
        self.layer_2 = tf.sigmoid(tf.matmul(layer_1, _W_prime) + b_prime)

        self.loss = - tf.reduce_sum(
            self.interact_matrix * tf.log(self.layer_2) + (1 - self.interact_matrix) * tf.log(1 - self.layer_2)) + \
            self.reg_rate * (tf.nn.l2_loss(_W) + tf.nn.l2_loss(_W_prime) +
                             tf.nn.l2_loss(b) + tf.nn.l2_loss(b_prime))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):
        self.train_data = self._data_process(train_data).transpose()
        self.neg_items = self._get_neg_items(train_data)
        self.num_training = self.num_item
        self.total_batch = int(self.num_training / self.batch_size)
        self.test_data = test_data
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(self.total_batch):
            start_time = time.time()
            if i == self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={
                self.corrupted_interact_matrix: self._get_corrupted_input(self.train_data[batch_set_idx, :],
                                                                          self.corruption_level),
                self.interact_matrix: self.train_data[batch_set_idx, :],
                self.item_id: batch_set_idx
                })
            if self.verbose and i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                if self.verbose:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.corrupted_interact_matrix: self.train_data,
                                                                     self.item_id: range(self.num_item)}).transpose()

        evaluate(self)

    def execute(self, train_data, test_data):
        self.prepare_data(train_data, test_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            self.train()
            if epoch % self.T == 0:
                print("Epoch: %04d; " % epoch, end='')
                self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return np.array(self.reconstruction[np.array(user_id), np.array(item_id)])

    @staticmethod
    def _data_process(data):
        return np.asmatrix(data)

    def _get_neg_items(self, data):
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = [k for k, i in enumerate(data[u]) if data[u][k] == 0]
            # print(neg_items[u])

        return neg_items

    @staticmethod
    def _get_corrupted_input(input_train_data, corruption_level):
        return np.random.binomial(n=1, p=1 - corruption_level) * input_train_data
