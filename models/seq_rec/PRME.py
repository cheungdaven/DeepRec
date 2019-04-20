#!/usr/bin/env python
"""Implementation of Caser.
Reference: Personalized Ranking Metric Embedding for Next New POI Recommendation, Shanshan Feng, IJCAI 2015.
"""

import tensorflow as tf
import time
import numpy as np
from utils.evaluation.SeqRecMetrics import *
import numpy as np

np.set_printoptions(threshold=np.inf)
__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class PRME():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=1e-2, epoch=5000, batch_size=1000,
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
        print("PRME.")

    def build_network(self, L, num_T, num_factor=100, num_neg=1):

        self.L = L
        self.num_T = num_T
        self.num_factor = num_factor
        self.num_neg = num_neg

        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_seq = tf.placeholder(dtype=tf.int32, shape=[None, L], name='item_seq')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None, self.num_T], name='item_id')
        self.item_id_test = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='item_id_test')
        self.neg_item_id = tf.placeholder(dtype=tf.int32, shape=[None, self.num_T * self.num_neg], name='item_id_neg')
        self.isTrain = tf.placeholder(tf.bool, shape=())


        print(np.shape(self.user_id))
        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=0.001))
        self.V = tf.Variable(tf.random_normal([self.num_user, num_factor * 1], stddev=0.001))
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=0.001))
        self.X = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=0.001))

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        self.user_specific_bias = tf.nn.embedding_lookup(self.V, self.user_id)



        self.target_prediction = self._distance(self.item_seq, user_latent_factor, self.item_id)
        self.negative_prediction = self._distance(self.item_seq, user_latent_factor, self.neg_item_id)
        self.test_prediction = self._distance(self.item_seq, user_latent_factor, self.item_id_test)

        # - tf.reduce_sum(tf.log(tf.sigmoid(self.target_prediction - self.negative_prediction)) )
        # - tf.reduce_mean(tf.log(tf.sigmoid(self.target_prediction) + 1e-10)) - tf.reduce_mean( tf.log( tf.sigmoid(1 - self.negative_prediction) + 1e-10))
        self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(- self.target_prediction + self.negative_prediction))) + 0.01 * (
            tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.X) + tf.nn.l2_loss(
                self.Q))  # + tf.losses.get_regularization_loss()
        '''
        self.loss =  tf.reduce_sum(tf.maximum(self.target_prediction - self.negative_prediction + 0.5, 0))   + self.reg_rate * ( tf.nn.l2_loss(self.P)   + tf.nn.l2_loss(self.V) +tf.nn.l2_loss(
                self.b)) + tf.losses.get_regularization_loss()


        self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.target_prediction - self.negative_prediction)) ) + self.reg_rate * (
            tf.nn.l2_loss(self.P) +    tf.nn.l2_loss(self.V) + tf.nn.l2_loss(
                self.W) + tf.nn.l2_loss(
                self.b)) + tf.losses.get_regularization_loss()


        '''
        norm_clip_value = 1
        self.clip_P = tf.assign(self.P, tf.clip_by_norm(self.P, norm_clip_value, axes=[1]))
        self.clip_Q = tf.assign(self.Q, tf.clip_by_norm(self.Q, norm_clip_value, axes=[1]))
        self.clip_V = tf.assign(self.V, tf.clip_by_norm(self.V, norm_clip_value, axes=[1]))
        self.clip_X = tf.assign(self.X, tf.clip_by_norm(self.X, norm_clip_value, axes=[1]))

        # self.loss = tf.reduce_sum(tf.square( 1 - self.target_prediction) ) + tf.reduce_sum(tf.square( self.negative_prediction)) + self.reg_rate * (
        #     tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) + tf.nn.l2_loss(self.V)+ tf.nn.l2_loss(self.W) + tf.nn.l2_loss(
        #         self.b)) + tf.losses.get_regularization_loss()

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)  # GradientDescentOptimizer

        return self



    def _distance(self, item_seq, user_latent_factor, item_id):

        item_latent_factor = tf.nn.embedding_lookup(self.Q, item_seq)

        out = tf.squeeze(item_latent_factor, 1)

        w_items = tf.nn.embedding_lookup(self.X, item_id)
        w_items_2 = tf.nn.embedding_lookup(self.Q, item_id)

        x_tmp = []
        for i in range(np.shape(w_items)[1]):
            x_tmp.append(out)
        x = tf.stack(x_tmp)
        print(np.shape(x))
        print(np.shape(w_items))
        x = tf.transpose(x, [1, 0, 2])


        u_tmp = []
        for i in range(np.shape(w_items)[1]):
            u_tmp.append(user_latent_factor)
        u = tf.stack(u_tmp)
        print(np.shape(u))
        u = tf.transpose(u, [1, 0, 2])

        res = 0.2 * tf.reduce_sum(tf.square(w_items - u), 2) + 0.8 * tf.reduce_sum(tf.square(x - w_items_2),2)

        return tf.squeeze(res)


    def prepare_data(self, train_data, test_data):
        self.sequences = train_data.sequences.sequences
        self.test_sequences = train_data.test_sequences.sequences
        self.targets = train_data.sequences.targets
        self.users = train_data.sequences.user_ids.reshape(-1, 1)
        all_items = set(np.arange(self.num_item - 1) + 1)
        self.all_items = all_items
        # print(all_items) # from 1 to 1679
        self.x = []
        for i, u in enumerate(self.users.squeeze()):
            tar = set([int(t) for t in self.targets[i]])
            seq = set([int(t) for t in self.sequences[i]])
            self.x.append(list(all_items - tar))
        self.x = np.array(self.x)

        self.test_data = dict()
        test = test_data.tocsr()

        for user, row in enumerate(test):
            self.test_data[user] = list(set(row.indices))

        if not self.neg_items:
            # all_items = set(np.arange(self.num_item - 1) + 1)
            train = train_data.tocsr()

            for user, row in enumerate(train):
                # print(user)
                # print(row.indices)
                # print(0 in row.indices)
                self.neg_items[user] = list(all_items - set(row.indices))
                # print(self.test_data[user][0] in self.neg_items[user])

    def execute(self, train_data, test_data):
        self.prepare_data(train_data, test_data)
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

        # print(self.test_sequences)
        idxs = np.random.permutation(
            self.num_training)  # shuffled ordering np.random.choice(self.num_training, self.num_training, replace=True) #

        sequences_random = [i.tolist() for i in list(self.sequences[idxs])]
        targets_random = list(self.targets[idxs])
        users_random = [i[0] for i in list(self.users[idxs])]
        self.x_random = list(self.x[idxs])
        item_random_neg = self._get_neg_items_sbpr(self.users.squeeze(), train_data, self.num_neg * self.num_T)

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

        # print(self.test_data)
        self.test_users = []
        for i in range(self.num_user):
            self.test_users.append(i)
        # print(self.test_users)
        evaluate(self)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        # print(user_id)
        # print(len(self.test_sequences))
        # print(self.test_sequences[user_id, :])
        # user_id_2 = [i+1 for i in user_id]
        item_id = [[i] for i in item_id]
        # print(len(item_id))

        return - self.sess.run([self.test_prediction], feed_dict={self.user_id: user_id,
                                                                  self.item_seq: self.test_sequences[user_id, :],
                                                                  self.item_id_test: item_id})[0]

    def _weight_variable(self, shape):
        initial = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)



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
