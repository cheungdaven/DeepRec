#!/usr/bin/env python
"""Implementation of Caser.
Reference: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18.
"""

import tensorflow as tf
import time
import numpy as np
from utils.evaluation.SeqRecMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class Caser():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=1e-6, epoch=500, batch_size=1000,
                 show_time=False, T=1, display_step=1000, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.verbose = verbose
        self.T = T
        self.display_step = display_step

        self.neg_items = dict()
        print("Caser.")

    def build_network(self, L, num_T, n_h=16, n_v=4, num_factor=150, num_neg=2):
        self.n_h = n_h
        self.n_v = n_v
        self.L = L
        self.num_T = num_T
        self.num_factor = num_factor
        self.num_neg = num_neg

        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_seq = tf.placeholder(dtype=tf.int32, shape=[None, L], name='item_seq')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None,  self.num_T], name='item_id')
        self.item_id_test = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='item_id_test')
        self.neg_item_id = tf.placeholder(dtype=tf.int32, shape=[None, self.num_T * self.num_neg], name='item_id_neg')
        self.isTrain = tf.placeholder(tf.bool, shape=())

        # self.y = tf.placeholder("float", [None], 'rating')
        print(np.shape(self.user_id))

        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=0.01))
        self.V = tf.Variable(tf.random_normal([self.num_user, num_factor * self.L], stddev=0.01))
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=0.01))

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        self.user_specific_bias =  tf.nn.embedding_lookup(self.V, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_seq)
        item_latent_factor_neg = tf.nn.embedding_lookup(self.Q, self.neg_item_id)
        self.W = tf.Variable(tf.random_normal([self.num_item, self.num_factor * 2 ], stddev=0.01))
        self.b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        # vertical conv layer
        # self.conv_v = tf.nn.conv2d(1, n_v, (L, 1))

        self.fc1_dim_v = self.n_v * self.num_factor
        self.fc1_dim_h = self.n_h * self.num_factor

        self.target_prediction = self._forward(item_latent_factor, user_latent_factor, self.item_id)
        self.negative_prediction = self._forward(item_latent_factor, user_latent_factor, self.neg_item_id)
        self.test_prediction = self._forward(item_latent_factor, user_latent_factor, self.item_id_test)



        self.loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.target_prediction) + 1e-10)) - tf.reduce_mean(
            tf.log(1 - tf.sigmoid(self.negative_prediction) + 1e-10)) + self.reg_rate * (
            tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) + tf.nn.l2_loss(self.V)+ tf.nn.l2_loss(self.W) + tf.nn.l2_loss(
                self.b)) + tf.losses.get_regularization_loss()


        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        return self


    def getUserParam(self, user_id):
        params = self.sess.run([self.user_emb], feed_dict={self.user_id: user_id, self.item_seq: self.test_sequences[user_id, :]})
        return params[0]

    def getItemParam(self, item_id):
        params = self.sess.run([self.w_items, self.b_items], feed_dict={self.item_id_test: item_id})
        return np.squeeze(params[0]), np.squeeze(params[1])

    def _forward(self, item_latent_factor, user_latent_factor, item_id):
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.L)]
        # self.conv_h = []
        #
        # for i in lengths:
        #     self.conv_h.append(tf.nn.conv2d(1, n_h, (i, num_factor)))


        # fully connected layer


        out, out_h, out_v = None, None, None
        if self.n_v:
            # print(tf.shape(tf.Variable(tf.random_normal(shape=[self.L, 1, 1, self.n_v], stddev=0.1))))
            # print(tf.expand_dims(item_latent_factor, 1))
            # out_v = tf.squeeze(tf.nn.conv2d(input=tf.expand_dims(item_latent_factor,3),
            #                                 filter=self._weight_variable(shape=[self.L, 1, 1, self.n_v]),
            #                                 strides=[1,1,1,1],
            #                                 padding='SAME'), 2)
            out_v = tf.squeeze(tf.layers.conv2d(inputs=tf.expand_dims(item_latent_factor, 3),
                                                filters=self.n_v,
                                                kernel_size=(self.L, 1),
                                                padding='valid',
                                                kernel_initializer=tf.random_normal_initializer,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                    scale=self.reg_rate),
                                                data_format='channels_last',
                                                reuse=tf.AUTO_REUSE,
                                                name="Convv"), 1)
            print(";;;;;;;;;;;;;;;;")

            out_v = tf.reshape(out_v, [-1, self.fc1_dim_v])
            print(np.shape(out_v))  # (?, 200)
        out_hs = list()
        if self.n_h:
            for i in lengths:
                conv_out = tf.nn.relu(tf.squeeze(tf.layers.conv2d(inputs=tf.expand_dims(item_latent_factor, 3),
                                                                  filters=self.n_h,
                                                                  kernel_size=(i, self.num_factor),
                                                                  padding='valid',
                                                                  kernel_initializer=tf.random_normal_initializer,
                                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                      scale=self.reg_rate),
                                                                  data_format='channels_last',
                                                                  reuse=tf.AUTO_REUSE,
                                                                  name="Convh" + str(i)), 2))
                # print(np.shape(conv_out))
                # print(np.shape(conv_out)[1])
                # print(tf.shape(conv_out))
                # conv_out = tf.transpose(conv_out, [0,2,1])
                pool_out = tf.squeeze(
                    tf.layers.max_pooling1d(conv_out, [np.shape(conv_out)[1]], data_format='channels_last',
                                            padding='valid', strides=1), 1)
                print(np.shape(pool_out))  # (?, 16)
                out_hs.append(pool_out)
            out_h = tf.concat(out_hs, axis=1)
            print(np.shape(out_h))  # (?, 80)

        out = tf.concat(values=[out_v, out_h], axis=1)
        print(np.shape(out))  # (?, 280)
        # fc1_dim_in = self.fc1_dim_h + self.fc1_dim_v

        # self.fc1 = tf.layers.dense(out, self.num_factor)
        # w and b are item specific


        self.w_items = tf.nn.embedding_lookup(self.W, item_id)
        self.b_items = tf.nn.embedding_lookup(self.b, item_id)

        z = tf.layers.dense(out, units=self.num_factor, activation=tf.nn.relu,
                            kernel_initializer=tf.random_normal_initializer,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                            name="full",
                            reuse=tf.AUTO_REUSE)
        print(np.shape(z))

        # x = tf.concat(values=[z, user_latent_factor], axis=1)
        # res = []
        # for i in range(np.shape(w_items)[1]):
        #     result = (x * w_items[:, i, :]) + b_items[:, i, 0]
        #     res.append(result)


        # x = tf.reshape(tf.tile(tf.concat(values=[z, user_latent_factor], axis=1),[1, np.shape(w_items)[1]]), [-1, np.shape(w_items)[1], self.num_factor * 2])

        self.user_emb = x = tf.concat(values=[z, user_latent_factor], axis=1)
        x_tmp = []
        for i in range(np.shape(self.w_items)[1]):
            x_tmp.append(x)
        x = tf.stack(x_tmp)
        print(np.shape(x))
        x = tf.transpose(x, [1, 0, 2])

        res = tf.reduce_sum(tf.multiply(x, self.w_items), 2) + self.b_items
        print("......")
        print(np.shape(res))

        return res

    def execute(self, train_data, test_data):
        self.sequences = train_data.sequences.sequences
        self.targets = train_data.sequences.targets
        self.users = train_data.sequences.user_ids.reshape(-1, 1)

        all_items = set(np.arange(self.num_item - 1) + 1)
        self.x = []
        for i, u in enumerate(self.users.squeeze()):
            tar = set([int(t) for t in self.targets[i]])
            seq = set([int(t) for t in self.sequences[i]])
            self.x.append(list(all_items - tar))
        self.x = np.array(self.x)
        if not self.neg_items:
            all_items = set(np.arange(self.num_item - 1) + 1)

            train = train_data.tocsr()

            for user, row in enumerate(train):
                self.neg_items[user] = list(all_items - set(row.indices))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            if self.verbose:
                print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch), end='')
                self.test(test_data)

    def train(self, train_data):

        # print(users)

        self.num_training = len(self.sequences)
        self.total_batch = int(self.num_training / self.batch_size)
        L, T = train_data.sequences.L, train_data.sequences.T
        self.test_sequences = train_data.test_sequences.sequences
        # print(self.test_sequences)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        sequences_random = [i.tolist() for i in list(self.sequences[idxs])]
        targets_random = list(self.targets[idxs])
        users_random = [i[0] for i in list(self.users[idxs])]
        self.x_random = list(self.x[idxs])
        item_random_neg =  self._get_neg_items( self.users.squeeze(), train_data, self.num_neg * self.num_T)
        # # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = users_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_seq = sequences_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = targets_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item_neg = item_random_neg[i * self.batch_size:(i + 1) * self.batch_size]
            # print(batch_item_neg)

            _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.user_id: batch_user,
                                                                            self.item_seq: batch_seq,
                                                                            self.item_id: batch_item,
                                                                            self.neg_item_id: batch_item_neg,
                                                                            self.isTrain: True})
            #
            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    #             print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        # print(test_data.user_map)
        self.test_data = dict()
        test = test_data.tocsr()

        for user, row in enumerate(test):
            self.test_data[user] = set(row.indices)
            # print(self.test_data[user])

        self.test_users = []
        for i in range(self.num_user):
            self.test_users.append(i)
        # print(self.test_users)
        evaluate_caser(self)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        # print(user_id)
        #print(self.test_sequences[user_id, :])
        # user_id_2 = [i+1 for i in user_id]
        item_id = [[i] for i in item_id]
        # print(len(item_id))

        return self.sess.run([self.test_prediction], feed_dict={self.user_id: user_id,
                                                                self.item_seq: self.test_sequences[user_id, :],
                                                                self.item_id_test: item_id})[0]

    def _weight_variable(self, shape):
        initial = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _get_neg_items(self, users, interactions, n):

        # users = users.squeeze()
        neg_items_samples = np.zeros((users.shape[0], n))

        # if not self.neg_items:
        #     all_items = set(np.arange(self.num_item - 1) + 1)
        #     train = interactions.tocsr()
        #
        #     for user, row in enumerate(train):
        #         self.neg_items[user] = list(all_items - set(row.indices))

        for i, u in enumerate(users):
            for j in range(n):
                x = self.neg_items[u]
                neg_items_samples[i, j] = x[np.random.randint(len(x))]

        return neg_items_samples

    def _get_neg_items_sbpr(self, users, interactions, n):
        # print("start sampling")
        # print(targets)
        # users = users.squeeze()
        neg_items_samples = np.zeros((users.shape[0], n))
        # all_items = None
        # if not self.neg_items:
        #     all_items = set(np.arange(self.num_item - 1) + 1)
        #     train = interactions.tocsr()
        #
        #     for user, row in enumerate(train):
        #         self.neg_items[user] = list(all_items - set(row.indices))
        print(len(users))
        for i, u in enumerate(users):
            for j in range(n):
                # print(int(targets[i][0]))
                neg_items_samples[i, j] = self.x_random[i][np.random.randint(len(self.x_random[i]))]
        # print("end sampling")
        return neg_items_samples




