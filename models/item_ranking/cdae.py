#!/usr/bin/env python
"""Implementation of CDAE.
Reference: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings
of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.
"""

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense, Add, Flatten, Reshape, Activation

from utils.evaluation.RankingMetrics import evaluate

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"

# TODO: need to move this into tf2_utils
class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, input_embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.input_embedding = input_embedding

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.input_embedding, inputs)


class CDAE(object):
    def __init__(self, num_user, num_item, learning_rate=0.01, reg_rate=0.01, epoch=500, batch_size=100,
                 verbose=False, t=1, display_step=1000):
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step
        self.loss_estimator = tf.keras.metrics.Mean(name='train_loss')
        self.loss_object = tf.keras.losses.BinaryCrossentropy()

        self.user_id = None
        self.corrupted_rating_matrix = None
        self.rating_matrix = None
        self.corruption_level = None
        self.layer_2 = None
        self.optimizer = None
        self.train_data = None
        self.neg_items = None
        self.num_training = None
        self.total_batch = None
        self.test_data = None
        self.test_users = None
        self.reconstruction = None

        self.optimizer = None
        self.model = None
        print("You are running CDAE.")

    def build_network(self, hidden_neuron=500, corruption_level=0):
        self.corruption_level = corruption_level

        corrupted_rating_matrix = Input(shape=(self.num_item,), dtype=tf.float32)
        rating_matrix = Input(shape=(self.num_item,), dtype=tf.float32)
        user_id = Input(shape=(1,), dtype=tf.int32)

        _V = tf.Variable(tf.random.normal([self.num_user, hidden_neuron], stddev=0.01))

        squeezed_user_id = Flatten()(user_id)
        user_latent_factor = EmbeddingLookup(_V)(squeezed_user_id)
        user_latent_factor = Reshape((_V.shape[-1],))(user_latent_factor)
        z_vector = Dense(hidden_neuron,
                         kernel_regularizer=regularizers.l2(self.reg_rate),
                         bias_regularizer=regularizers.l2(self.reg_rate)
                         )(corrupted_rating_matrix)

        z_vector = Add()([z_vector, user_latent_factor])
        z_vector = Activation(tf.nn.sigmoid)(z_vector)

        recon_vector = Dense(self.num_item, activation='sigmoid',
                             kernel_regularizer=regularizers.l2(self.reg_rate),
                             bias_regularizer=regularizers.l2(self.reg_rate)
                             )(z_vector)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = Model(inputs=[corrupted_rating_matrix, rating_matrix, user_id],
                           outputs=recon_vector)

    def prepare_data(self, train_data, test_data):
        self.train_data = self._data_process(train_data)
        self.neg_items = self._get_neg_items(train_data)
        self.num_training = self.num_user
        self.total_batch = int(self.num_training / self.batch_size)
        self.test_data = test_data
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")

    @tf.function
    def train_op(self, corrupted_rating_matrix, rating_matrix, user_id):
        with tf.GradientTape() as tape:
            recon_vector = self.model([corrupted_rating_matrix, rating_matrix, user_id])
            loss = self.loss_object(rating_matrix, recon_vector)
        gradient_of_model = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_of_model, self.model.trainable_variables))
        self.loss_estimator(loss)

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(self.total_batch):
            start_time = time.time()
            if i == self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            expanded_batch_set_idx = np.expand_dims(np.array(batch_set_idx), -1)
            self.train_op(
                self._get_corrupted_input(self.train_data[batch_set_idx, :], self.corruption_level),
                np.array(self.train_data[batch_set_idx, :], np.float32),
                expanded_batch_set_idx)

            if self.verbose and i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, self.loss_estimator.result()))
                if self.verbose:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        reconstruction = self.model([self.train_data, self.train_data, np.expand_dims(range(self.num_user), -1)])
        self.reconstruction = np.array(reconstruction)
        evaluate(self)

    def execute(self, train_data, test_data):
        self.prepare_data(train_data, test_data)

        for epoch in range(self.epochs):
            self.train()
            if epoch % self.T == 0:
                print("Epoch: %04d; " % epoch, end='')
                self.test()

    def save(self, path):
        tf.saved_model.save(self.model, path)

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
        return np.array(np.random.binomial(n=1, p=1 - corruption_level) * input_train_data)


class ICDAE(object):
    """
    Based on CDAE and I-AutoRec, I designed the following item based CDAE, it seems to perform better than CDAE
    slightly.
    """

    def __init__(self, num_user, num_item, learning_rate=0.01, reg_rate=0.01, epoch=500, batch_size=300,
                 verbose=False, t=2, display_step=1000):
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step
        self.loss_estimator = tf.keras.metrics.Mean(name='train_loss')
        self.loss_object = tf.keras.losses.BinaryCrossentropy()

        self.corrupted_interact_matrix = None
        self.interact_matrix = None
        self.corruption_level = None
        self.layer_2 = None
        self.optimizer = None
        self.train_data = None
        self.neg_items = None
        self.num_training = None
        self.total_batch = None
        self.test_data = None
        self.test_users = None
        self.reconstruction = None

        self.optimizer = None
        self.model = None
        print("Item based CDAE.")

    def build_network(self, hidden_neuron=500, corruption_level=0):
        # self.corrupted_interact_matrix = tf.placeholder(dtype=tf.float32, shape=[None, self.num_user])
        # self.interact_matrix = tf.placeholder(dtype=tf.float32, shape=[None, self.num_user])
        # self.item_id = tf.placeholder(dtype=tf.int32, shape=[None])
        self.corruption_level = corruption_level
        corrupted_rating_matrix = Input(shape=(self.num_user,), dtype=tf.float32)
        _V = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))

        z_vector = Dense(hidden_neuron,
                         activation='sigmoid',
                         kernel_regularizer=regularizers.l2(self.reg_rate),
                         bias_regularizer=regularizers.l2(self.reg_rate))(corrupted_rating_matrix)

        recon_vector = Dense(self.num_user, activation='sigmoid',
                             kernel_regularizer=regularizers.l2(self.reg_rate),
                             bias_regularizer=regularizers.l2(self.reg_rate))(z_vector)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = Model(inputs=corrupted_rating_matrix,
                           outputs=recon_vector)

    def prepare_data(self, train_data, test_data):
        self.train_data = self._data_process(train_data).transpose()
        self.neg_items = self._get_neg_items(train_data)
        self.num_training = self.num_item
        self.total_batch = int(self.num_training / self.batch_size)
        self.test_data = test_data
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")

    @tf.function
    def train_op(self, corrupted_rating_matrix, rating_matrix):
        with tf.GradientTape() as tape:
            recon_vector = self.model([corrupted_rating_matrix])
            loss = self.loss_object(rating_matrix, recon_vector)
        gradient_of_model = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_of_model, self.model.trainable_variables))
        self.loss_estimator(loss)

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(self.total_batch):
            start_time = time.time()
            if i == self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            self.train_op(
                self._get_corrupted_input(np.array(self.train_data[batch_set_idx, :], dtype=np.float32), self.corruption_level),
                np.array(self.train_data[batch_set_idx, :], np.float32)
                )

            if self.verbose and i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, self.loss_estimator.result()))
                if self.verbose:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        reconstruction = self.model([self.train_data, self.train_data])
        self.reconstruction = np.array(reconstruction)
        evaluate(self)

    def execute(self, train_data, test_data):
        self.prepare_data(train_data, test_data)

        for epoch in range(self.epochs):
            self.train()
            if epoch % self.T == 0:
                print("Epoch: %04d; " % epoch, end='')
                self.test()

    def save(self, path):
        tf.saved_model.save(self.model, path)

    def predict(self, user_id, item_id):
        return np.array(self.reconstruction[np.array(item_id), np.array(user_id)])

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
