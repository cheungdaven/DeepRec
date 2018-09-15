#!/usr/bin/env python
"""Implementation of Item based AutoRec and user based AutoRec.
Reference: Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th International Conference on World Wide Web. ACM, 2015.
"""

import tensorflow as tf
import time
import numpy as np
import scipy

from utils.evaluation.RatingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class IAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=500,
                 verbose=False, T=3, display_step=1000):
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
        print("IAutoRec.")

    def build_network(self, hidden_neuron=500):

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.keep_rate_net: 0.95
                                               })
            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        print("RMSE:" + str(RMSE(error, len(test_set))) + "; MAE:" + str(MAE(error_mae, len(test_set))))

    def execute(self, train_data, test_data):
        self.train_data = self._data_process(train_data)
        self.train_data_mask = scipy.sign(self.train_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            if self.verbose:
                print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch), end='')
                self.test(test_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i))
        return output


class UAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=200,
                 verbose=False, T=3, display_step=1000):
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
        print("UAutoRec.")

    def build_network(self, hidden_neuron=500):

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_item], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_item, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):
        self.num_training = self.num_user
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx]
                                               })
            if self.verbose and i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                if self.verbose:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        print("RMSE:" + str(RMSE(error, len(test_set))) + "; MAE:" + str(MAE(error_mae, len(test_set))))

    def execute(self, train_data, test_data):
        self.train_data = self._data_process(train_data.transpose())
        self.train_data_mask = scipy.sign(self.train_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            self.train(train_data)
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch), end='')
                self.test(test_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[item_id, user_id]

    def _data_process(self, data):
        output = np.zeros((self.num_item, self.num_user))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[i, u] = data.get((i, u))
        return output
