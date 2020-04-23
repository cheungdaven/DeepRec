#!/usr/bin/env python
"""Implementation of Neural Collaborative Filtering.
Reference: He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference
on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
"""

import tensorflow as tf
from tensorflow.keras import Input, regularizers, Model
from tensorflow.keras.layers import Flatten, Embedding, Multiply, concatenate, Dense, Lambda
import time
import random

from utils.evaluation.RankingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class NeuMF(object):
    def __init__(self, num_user, num_item, learning_rate=0.5, reg_rate=0.01, epoch=500, batch_size=256,
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

        self.num_neg_sample = None
        self.user_id = None
        self.item_id = None
        self.y = None
        self.P = None
        self.Q = None
        self.mlp_P = None
        self.mlp_Q = None
        self.pred_y = None
        self.loss_estimator = tf.keras.metrics.Mean(name='train_loss')
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = None

        self.test_data = None
        self.user = None
        self.item = None
        self.label = None
        self.neg_items = None
        self.test_users = None

        self.num_training = None
        self.total_batch = None
        print("You are running NeuMF.")

    def build_network(self, num_factor=10, num_factor_mlp=64, hidden_dimension=10, num_neg_sample=30):
        self.num_neg_sample = num_neg_sample
        user_id = Input(shape=(1,), dtype=tf.int32, name='user_id')
        item_id = Input(shape=(1,), dtype=tf.int32, name='item_id')

        P = Embedding(input_dim=self.num_user, output_dim=num_factor, name='mf_embedding_user',
                                      embeddings_initializer='normal', input_length=1)
        Q = Embedding(input_dim=self.num_item, output_dim=num_factor, name='mf_embedding_item',
                                      embeddings_initializer='normal', input_length=1)

        mlp_P = Embedding(input_dim=self.num_user, output_dim=num_factor_mlp, name="mlp_embedding_user",
                                       embeddings_initializer='normal', input_length=1)
        mlp_Q = Embedding(input_dim=self.num_item, output_dim=num_factor_mlp, name='mlp_embedding_item',
                                       embeddings_initializer='normal', input_length=1)

        flatten_user_id = Flatten()(user_id)
        flatten_item_id = Flatten()(item_id)

        user_latent_factor = Flatten()(P(flatten_user_id))
        item_latent_factor = Flatten()(Q(flatten_item_id))

        mlp_user_latent_factor = Flatten()(mlp_P(flatten_user_id))
        mlp_item_latent_factor = Flatten()(mlp_Q(flatten_item_id))

        _GMF = Multiply()([user_latent_factor, item_latent_factor])

        layer_1 = concatenate([mlp_item_latent_factor, mlp_user_latent_factor])
        layer_1 = Dense(num_factor_mlp, activation='relu',
                        kernel_regularizer=regularizers.l2(self.reg_rate)
                        )(layer_1)

        layer_2 = Dense(hidden_dimension * 8, activation='relu',
                        kernel_regularizer=regularizers.l2(self.reg_rate)
                        )(layer_1)

        layer_3 = Dense(hidden_dimension * 4,
                        activation='relu',
                        kernel_regularizer=regularizers.l2(self.reg_rate)
                        )(layer_2)

        layer_4 = Dense(hidden_dimension * 2,
                        activation='relu',
                        kernel_regularizer=regularizers.l2(self.reg_rate)
                        )(layer_3)

        _MLP = Dense(hidden_dimension,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(self.reg_rate)
                     )(layer_4)

        pred_y = Dense(1, activation='sigmoid')(concatenate([_GMF, _MLP]))
        self.model = Model(inputs=[user_id, item_id],
                            outputs=pred_y)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def prepare_data(self, train_data, test_data):
        """
        You must prepare the data before train and test the model.

        :param train_data:
        :param test_data:
        :return:
        """
        t = train_data.tocoo()
        self.user = list(t.row.reshape(-1))
        self.item = list(t.col.reshape(-1))
        self.label = list(t.data)
        self.test_data = test_data

        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])

        print("data preparation finished.")
        return self

    @tf.function
    def train_op(self, batch_user, batch_item, batch_label):
        with tf.GradientTape() as tape:
            pred_y = self.model([batch_user, batch_item])
            loss = self.loss_object(batch_label, pred_y)
        gradient_of_model = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_of_model, self.model.trainable_variables))
        self.loss_estimator(loss)

    def train(self):
        item_temp = self.item[:]
        user_temp = self.user[:]
        labels_temp = self.label[:]

        user_append = []
        item_append = []
        values_append = []
        for u in self.user:
            list_of_random_items = random.sample(self.neg_items[u], self.num_neg_sample)
            user_append += [u] * self.num_neg_sample
            item_append += list_of_random_items
            values_append += [0] * self.num_neg_sample

        item_temp += item_append
        user_temp += user_append
        labels_temp += values_append

        self.num_training = len(item_temp)
        self.total_batch = int(self.num_training / self.batch_size)
        # print(self.total_batch)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = np.array(user_temp)[idxs]
        item_random = np.array(item_temp)[idxs]
        labels_random = np.array(labels_temp)[idxs]

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_label = labels_random[i * self.batch_size:(i + 1) * self.batch_size]

            batch_user = np.expand_dims(batch_user, -1)
            batch_item = np.expand_dims(batch_item, -1)
            batch_label = np.expand_dims(batch_label, -1)

            self.train_op(batch_user, batch_item, batch_label)

            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, self.loss_estimator.result()))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
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
        user_id = np.expand_dims(np.array(user_id), -1)
        item_id = np.expand_dims(np.array(item_id), -1)
        return np.array(self.model([user_id, item_id]))

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items
