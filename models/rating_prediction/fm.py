#!/usr/bin/env python
"""Implementation of Matrix Factorization with tensorflow.
Reference: Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009).
Orginal Implementation:
"""

import tensorflow as tf
import time
from sklearn.metrics import mean_squared_error
import math

from utils.evaluation.RatingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"

class FM():

    def __init__(self, sess, num_user, num_item, learning_rate = 0.05, reg_rate = 0.01, epoch = 500, batch_size = 128, show_time = False, T =2, display_step= 1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("FM.")


    def build_network(self, feature_M, num_factor = 64):


        # model dependent arguments
        self.train_features = tf.placeholder(tf.int32, shape=[None, None])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.dropout_keep = tf.placeholder(tf.float32)

        self.feature_embeddings = tf.Variable(tf.random_normal([feature_M, num_factor], mean=0.0, stddev=0.01))

        self.feature_bias = tf.Variable(tf.random_uniform([feature_M, 1], 0.0, 0.0))
        self.bias = tf.Variable(tf.constant(0.0))


        nonzero_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, self.train_features)

        self.summed_features_embedding = tf.reduce_sum(nonzero_embeddings, 1)
        self.squared_summed_features_embedding = tf.square(self.summed_features_embedding)
        self.squared_features_embedding = tf.square(nonzero_embeddings)
        self.summed_squared_features_embedding = tf.reduce_sum(self.squared_features_embedding, 1)

        self.FM = 0.5 * tf.subtract(self.summed_squared_features_embedding, self.squared_summed_features_embedding)
        # if batch_norm:
        #     self.FM = self
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep)

        bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)
        self.f_b = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_bias, self.train_features), 1)
        b = self.bias * tf.ones_like(self.y)
        self.pred_rating = tf.add_n([bilinear, self.f_b, b])

        self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.pred_rating)) \
                    + tf.contrib.layers.l2_regularizer(self.reg_rate)(self.feature_embeddings)


        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):


        print("data preparation finished.")
        return self


    def train(self, train_data):
        self.num_training = len(train_data['Y'])
        total_batch = int( self.num_training/ self.batch_size)

        rng_state = np.random.get_state()
        np.random.shuffle(train_data['Y'])
        np.random.set_state(rng_state)
        np.random.shuffle(train_data['X'])
        # train
        for i in range(total_batch):
            start_time = time.time()
            batch_y = train_data['Y'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_x = train_data['X'][i * self.batch_size:(i + 1) * self.batch_size]
            print(batch_x)
            loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict={self.train_features: batch_x,
                                                                              self.y: batch_y,
                                                                              self.dropout_keep:0.5})
            if i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                if self.show_time:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        num_example = len(test_data['Y'])
        feed_dict = {self.train_features: test_data['X'], self.y: test_data['Y'],self.dropout_keep: 1.0}
        predictions = self.sess.run((self.pred_rating), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(test_data['Y'], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

        print("RMSE:" + str(RMSE))

    def execute(self, train_data, test_data):

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            if (epoch) % self.T == 0:
                self.test(test_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

