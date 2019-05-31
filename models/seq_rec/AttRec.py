#!/usr/bin/env python
"""Implementation of Caser.
Reference: Next Item Recommendation with Self-Attentive Metric Learning, Shuai Zhang etc. , AAAI workshop'18.
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


class AttRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.05, reg_rate=1e-2, epoch=5000, batch_size=1000,
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
        print("AttSeqRec.")

    def build_network(self, L, num_T, num_factor=150, num_neg=1):

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

        # self.y = tf.placeholder("float", [None], 'rating')
        print(np.shape(self.user_id))

        # initializer = tf.contrib.layers.xavier_initializer()
        # self.P = tf.Variable(initializer([self.num_user, num_factor] ))
        # self.V = tf.Variable(initializer([self.num_user, num_factor * 1] ))
        # self.Q = tf.Variable(initializer([self.num_item, num_factor] ))
        # self.X = tf.Variable(initializer([self.num_item, num_factor] ))

        self.P = tf.Variable(tf.truncated_normal([self.num_user, num_factor], stddev=0.001))
        self.V = tf.Variable(tf.truncated_normal([self.num_user, num_factor * 1], stddev=0.001))
        self.Q = tf.Variable(tf.truncated_normal([self.num_item, num_factor], stddev=0.001))
        self.X = tf.Variable(tf.truncated_normal([self.num_item, num_factor], stddev=0.001))
        self.A = tf.Variable(tf.truncated_normal([self.num_item, num_factor], stddev=0.001))


        item_latent_factor_neg = tf.nn.embedding_lookup(self.Q, self.neg_item_id)
        self.W = tf.Variable(tf.random_normal([self.num_item, self.num_factor * (1 + 1)], stddev=0.01))
        self.b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        # vertical conv layer
        # self.conv_v = tf.nn.conv2d(1, n_v, (L, 1))



        self.target_prediction = self._distance_self_attention(self.item_seq, self.user_id, self.item_id)
        self.negative_prediction = self._distance_self_attention(self.item_seq, self.user_id, self.neg_item_id)
        self.test_prediction = self._distance_self_attention(self.item_seq, self.user_id, self.item_id_test, isTrain=False)

        self.user_param = self.user_latent_factor
        self.user_param_2 = self.user_specific_bias

        self.seq_param = self.out
        self.seq_weight = self.weights
        self.item_param_1, self.item_param_2 = self._getItemParam(self.item_id_test)

        # - tf.reduce_sum(tf.log(tf.sigmoid(- self.target_prediction + self.negative_prediction)) )
        # - tf.reduce_mean(tf.log(tf.sigmoid(self.target_prediction) + 1e-10)) - tf.reduce_mean( tf.log( tf.sigmoid(1 - self.negative_prediction) + 1e-10))
        # tf.reduce_mean(tf.square(1 - self.negative_prediction)) + tf.reduce_mean(tf.square(self.target_prediction))
        # tf.reduce_sum(tf.maximum(self.target_prediction - self.negative_prediction + 0.5, 0))



        self.loss = tf.reduce_sum(tf.maximum(self.target_prediction - self.negative_prediction + 0.5, 0)) \
                    + tf.losses.get_regularization_loss() + 0.001 * (
        tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.V) + tf.nn.l2_loss(self.X) + tf.nn.l2_loss(self.Q))


        norm_clip_value = 1
        self.clip_P = tf.assign(self.P, tf.clip_by_norm(self.P, norm_clip_value, axes=[1]))
        self.clip_Q = tf.assign(self.Q, tf.clip_by_norm(self.Q, norm_clip_value, axes=[1]))
        self.clip_V = tf.assign(self.V, tf.clip_by_norm(self.V, norm_clip_value, axes=[1]))
        self.clip_X = tf.assign(self.X, tf.clip_by_norm(self.X, norm_clip_value, axes=[1]))



        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate, initial_accumulator_value=0.05).minimize(self.loss)  # GradientDescentOptimizer

        return self



    def getUserParam(self, user_id):
        params = self.sess.run([self.user_param, self.seq_param,  self.user_param_2, self.seq_weight], feed_dict={self.user_id: user_id,
                                                           self.item_seq: self.test_sequences[user_id, :]})
        return params[0], params[1], params[2], params[3]

    def getItemParam(self, item_id):
        params = self.sess.run([self.item_param_1, self.item_param_2,  self.bias_item], feed_dict={self.item_id_test: item_id})
        return np.squeeze(params[0]), np.squeeze(params[1]), params[2]

    def _getItemParam(self, item_id):
        w_items = tf.nn.embedding_lookup(self.X, item_id)
        w_items_2 = tf.nn.embedding_lookup(self.Q, item_id)
        return w_items, w_items_2



    def _distance_self_attention(self, item_seq, user_id, item_id, isTrain=True):
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.L)]

        out, out_h, out_v = None, None, None
        # print(np.shape(item_seq)[1])


        # item_latent_factor = self.add_timing_signal(item_latent_factor)

        item_latent_factor =  tf.nn.embedding_lookup(self.Q, item_seq) #+ self.user_specific_bias


        item_latent_factor_2 = tf.nn.embedding_lookup(self.X, item_seq)

        query = key = value = self.add_timing_signal(item_latent_factor)

        if isTrain:
            query = tf.layers.dense(inputs=query, name="linear_project", units=self.num_factor, activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
                                    )
            #query = tf.layers.dropout(query, rate=0.0)
            key = tf.layers.dense(inputs=key, name="linear_project", units=self.num_factor, activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                  )
            #key = tf.layers.dropout(key, rate=0.3)
        else:
            query = tf.layers.dense(inputs=query, name="linear_project", units=self.num_factor, activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
                                    )
            key = tf.layers.dense(inputs=key, name="linear_project", units=self.num_factor, activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
                                  )

        logits = tf.matmul(query, key, transpose_b=True) / np.sqrt(self.num_factor)

        print(np.shape(logits))
        weights = tf.nn.softmax(logits, dim=-1, name="attention_weights")
        mask = tf.ones([self.L, self.L])
        mask = tf.matrix_set_diag(mask, tf.zeros([self.L]))
        weights = weights * mask

        out =   tf.matmul(weights, item_latent_factor )
        self.weights = weights

        print(np.shape(item_latent_factor))
        self.out = tf.reduce_mean(out, 1)
        print(np.shape(self.out))


        w_items =  tf.nn.embedding_lookup(self.X, item_id)
        w_items_2 = tf.nn.embedding_lookup(self.Q, item_id)
        w_items_3 = tf.nn.embedding_lookup(self.V, user_id)#tf.nn.embedding_lookup(self.A, item_id)
        self.bias_item = tf.nn.embedding_lookup(self.b, item_id)


        x_tmp = []
        for i in range(np.shape(item_id)[1]):
            x_tmp.append(self.out)
        x = tf.stack(x_tmp)
        print(np.shape(x))
        print(np.shape(w_items))
        x = tf.transpose(x, [1, 0, 2])


        self.user_latent_factor = tf.nn.embedding_lookup(self.P, user_id)


        u_tmp = []
        for i in range(np.shape(item_id)[1]):
            u_tmp.append(self.user_latent_factor)
        u = tf.stack(u_tmp)
        print(np.shape(u))
        u = tf.transpose(u, [1, 0, 2])

        self.user_specific_bias = tf.nn.embedding_lookup(self.V, user_id)
        u_tmp_2 = []
        for i in range(np.shape(item_id)[1]):
            u_tmp_2.append(self.user_specific_bias)
        u_2 = tf.stack(u_tmp_2)
        print(np.shape(u_2))
        u_2 = tf.transpose(u_2, [1, 0, 2])


        self.alpha = 0.2
        if isTrain:
            res = self.alpha * tf.reduce_sum(tf.nn.dropout(tf.square(w_items - u), 1), 2) + (1-self.alpha) * tf.reduce_sum(tf.nn.dropout(tf.square(x -w_items_2  ),1),2)   #+ 0.1 * tf.reduce_sum(tf.square(x - u), 2)
        else:
            res = self.alpha * tf.reduce_sum(tf.square(w_items - u), 2) + (1 - self.alpha) * tf.reduce_sum(
                tf.square(x - w_items_2 ), 2)

        print(np.shape(res))
        return tf.squeeze(res)


    def _distance_multihead(self, item_seq, user_latent_factor, item_id):
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.L)]

        out, out_h, out_v = None, None, None

        # item_latent_factor = self.add_timing_signal(item_latent_factor)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, item_seq)
        item_latent_factor_2 = tf.nn.embedding_lookup(self.X, item_seq)

        query = key = self.add_timing_signal(item_latent_factor)

        out = self.multihead_attention(queries=query, keys=key, value=item_latent_factor, reuse=tf.AUTO_REUSE)
        out = tf.reduce_mean(out, 1)

        query_2 = key_2 = out
        query_2 = tf.layers.dense(inputs=query_2, name="linear_project1", units=self.num_factor, activation=None,
                                  use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
                                  )
        key_2 = tf.layers.dense(inputs=key_2, name="linear_project1", units=self.num_factor, activation=None,
                                use_bias=False,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
                                )
        # value = tf.layers.dense(inputs= key, name="linear_project", units = seq_len_item, activation = None,  kernel_initializer=tf.random_normal_initializer, reuse=True)
        # b =  tf.Variable(tf.random_normal([seq_len_user], stddev=1))
        logits_2 = tf.matmul(query_2, key_2, transpose_b=True) / np.sqrt(self.num_factor)
        weights_2 = tf.nn.softmax(logits_2, name="attention_weights1")
        mask_2 = tf.ones([self.L, self.L])
        mask_2 = tf.matrix_set_diag(mask_2, tf.zeros([self.L]))
        weights_2 = weights_2 * mask_2
        out_2 = tf.reduce_mean(tf.matmul(weights_2, out) , 1)

        print("--------------")
        print(np.shape(out))
        print(np.shape(out_2))




        w_items = tf.nn.embedding_lookup(self.X, item_id)
        w_items_2 = tf.nn.embedding_lookup(self.Q, item_id)
        b_items = tf.nn.embedding_lookup(self.b, item_id)
        item_specific_bias = tf.nn.embedding_lookup(self.X, item_id)


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

        # res = tf.reduce_sum(tf.multiply(x, w_items), 2) + b_items
        res = 0.2 * tf.reduce_sum(tf.square(w_items - u), 2) + 0.8 * tf.reduce_sum(tf.square(x- w_items_2),2)   # + 0.1 * tf.reduce_sum(tf.square(x - u), 2)


        print(np.shape(res))
        return tf.squeeze(res)

    def execute(self, train_data, test_data):

        self.prepare_data(train_data, test_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            if self.verbose:
                print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            if (epoch) % self.T == 0 and epoch >= 5:
                print("Epoch: %04d; " % (epoch), end='')
                self.test(test_data)


    def prepare_data(self, train_data, test_data):
        self.sequences = train_data.sequences.sequences
        # print(self.sequences)
        self.targets = train_data.sequences.targets
        self.users = train_data.sequences.user_ids.reshape(-1, 1)
        all_items = set(np.arange(self.num_item - 1) + 1)
        self.all_items = all_items
        # print(all_items) # from 1 to 1679


        self.test_data = dict()
        test = test_data.tocsr()

        for user, row in enumerate(test):
            self.test_data[user] = list(set(row.indices))

        self.x = []
        for i, u in enumerate(self.users.squeeze()):
            tar = set([int(t) for t in self.targets[i]])
            # print(tar)
            seq = set([int(t) for t in self.sequences[i]])
            self.x.append(list(all_items - tar))
            # print(self.test_data[u][0] in  self.x[i])
        self.x = np.array(self.x)

        if not self.neg_items:
            # all_items = set(np.arange(self.num_item - 1) + 1)
            train = train_data.tocsr()

            for user, row in enumerate(train):
                # print(user)
                # print(row.indices)
                # print(0 in row.indices)
                self.neg_items[user] = list(all_items - set(row.indices))
                # print(self.test_data[user][0] in self.neg_items[user])
        print("Data Preparation Finish.")

    def train(self, train_data):

        # print(users)

        self.num_training = len(self.sequences)
        self.total_batch = int(self.num_training / self.batch_size)
        L, T = train_data.sequences.L, train_data.sequences.T
        self.test_sequences = train_data.test_sequences.sequences
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
            #, self.clip_P, self.clip_Q, self.clip_X
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
        evaluate1(self)

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

        return -self.sess.run([self.test_prediction], feed_dict={self.user_id: user_id,
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
        print(users.shape[0])
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

    def add_timing_signal(self, x, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.
        Each channel of the input Tensor is incremented by a sinusoid of a
        different frequency and phase.
        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.
        The use of relative position is possible because sin(x+y) and cos(x+y) can
        be experessed in terms of y, sin(x) and cos(x).
        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels / 2. For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.
        Args:
            x: a Tensor with shape [batch, length, channels]
            min_timescale: a float
            max_timescale: a float
        Returns:
            a Tensor the same shape as x.
        """
        with tf.name_scope("add_timing_signal", values=[x]):
            length = tf.shape(x)[1]
            channels = tf.shape(x)[2]
            position = tf.to_float(tf.range(length))
            num_timescales = channels // 2

            log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
            )
            inv_timescales = min_timescale * tf.exp(
                tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
            )

            scaled_time = (tf.expand_dims(position, 1) *
                           tf.expand_dims(inv_timescales, 0))
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
            signal = tf.reshape(signal, [1, length, channels])

            return x + signal

    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self, queries, keys, value, num_units=None, num_heads=2, dropout_rate=0, is_training=True, causality=False, scope="multihead_attention", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]


            Q = tf.layers.dense(queries, num_units, name="project_q", activation=tf.nn.relu,
                                use_bias=False,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate)) # (batch size, sequence length, dim)
            K = tf.layers.dense(keys, num_units, name="project_k", activation=tf.nn.relu,
                                use_bias=False,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
            V = tf.layers.dense(value, num_units, name="project_v",activation=tf.nn.relu,
                                use_bias=False,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2),axis=0) #( h * batch size, seq len, dim/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2),axis=0)
            V_ = tf.concat(tf.split(value, num_heads, axis=2),axis=0)


            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

            # scale
            outputs = outputs / ( K_.get_shape().as_list()[-1] ** 0.5)

            # key masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (batch size, seq len)
            key_masks = tf.tile(key_masks, [num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [ 1, tf.shape(queries)[1], 1])

            paddings = tf.ones_like(outputs) * (-2**32 + 1)

            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)


            if causality:
                diag_vals = tf.ones_like(outputs[0,:,:])
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

                paddings = tf.ones_like(masks) * ( -2**32 +1 )
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

            outputs = tf.nn.sigmoid(outputs)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1,1, tf.shape(keys)[1]])
            outputs *= query_masks

            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            outputs = tf.matmul(outputs, V_)

            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

            #outputs += value

            #outputs = self.normalize(outputs, reuse=tf.AUTO_REUSE)
        return outputs


    def feedforward(self, inputs,
                    num_units=[400, 100],
                    scope="multihead_attention",
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.normalize(outputs, reuse=tf.AUTO_REUSE)

        return outputs

    def add_timing_signal_nd(self, x, min_timescale=1.0, max_timescale=1.0e4):
        """ Adds a bunch of sinusoids of different frequencies to a Tensor.
            Each channel of the input Tensor is incremented by a sinusoid of a
            different frequency and phase in one of the positional dimensions.
            This allows attention to learn to use absolute and relative positions.
            Timing signals should be added to some precursors of both the query and
            the memory inputs to attention.
            The use of relative position is possible because sin(a+b) and cos(a+b)
            can be experessed in terms of b, sin(a) and cos(a).
            x is a Tensor with n "positional" dimensions, e.g. one dimension for a
            sequence or two dimensions for an image
            We use a geometric sequence of timescales starting with min_timescale
            and ending with max_timescale.  The number of different timescales is
            equal to channels // (n * 2). For each timescale, we generate the two
            sinusoidal signals sin(timestep/timescale) and cos(timestep/timescale).
            All of these sinusoids are concatenated in the channels dimension.
            Args:
                x: a Tensor with shape [batch, d1 ... dn, channels]
                min_timescale: a float
                max_timescale: a float
            Returns:
                a Tensor the same shape as x.
        """
        static_shape = x.get_shape().as_list()
        num_dims = len(static_shape) - 2
        channels = tf.shape(x)[-1]
        num_timescales = channels // (num_dims * 2)
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
              tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )
        for dim in range(num_dims):
            length = tf.shape(x)[dim + 1]
            position = tf.to_float(tf.range(length))
            scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
            for _ in range(1 + dim):
                signal = tf.expand_dims(signal, 0)
            for _ in range(num_dims - 1 - dim):
                signal = tf.expand_dims(signal, -2)
            x += signal

        return x
