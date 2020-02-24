#!/usr/bin/env python
"""Implementation of Bayesain Personalized Ranking Model.
Reference: Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of
the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009..
"""

import time

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda

from utils.evaluation.RankingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, input_embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.input_embedding = input_embedding

    # def build(self, input_shape):
    #     self.embeddings = self.add_weight(
    #         shape=(self.input_dim, self.output_dim),
    #         initializer='random_normal',
    #         dtype='float32')

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.input_embedding, inputs)


class BPRMF(object):
    def __init__(self, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=1024,
                 verbose=False, t=5, display_step=1000):
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step

        self.user_id = Input(shape=(1,), dtype=tf.int32, name='user_id')
        self.item_id = Input(shape=(1,), dtype=tf.int32, name='item_id')
        self.neg_item_id = Input(shape=(1,), dtype=tf.int32, name='neg_item_id')
        self.P = None
        self.Q = None
        self.pred_y = None
        self.pred_y_neg = None
        self.loss = None
        self.loss_estimator = tf.keras.metrics.Mean(name='train_loss')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.test_data = None
        self.user = None
        self.item = None
        self.neg_items = None
        self.test_users = None

        self.num_training = None
        self.total_batch = None
        print("You are running BPRMF.")

    def build_network(self, num_factor=30):
        self.P = tf.Variable(tf.random.normal([self.num_user, num_factor], stddev=0.01))
        self.Q = tf.Variable(tf.random.normal([self.num_item, num_factor], stddev=0.01))
        # user_id = tf.squeeze(self.user_id)
        # item_id = tf.squeeze(self.item_id)
        # neg_item_id = tf.squeeze(self.neg_item_id)
        user_id = Lambda(lambda x: tf.squeeze(x))(self.user_id)
        item_id = Lambda(lambda x: tf.squeeze(x))(self.item_id)
        neg_item_id = Lambda(lambda x: tf.squeeze(x))(self.neg_item_id)

        user_latent_factor = EmbeddingLookup(self.P)(user_id)
        item_latent_factor = EmbeddingLookup(self.Q)(item_id)
        neg_item_latent_factor = EmbeddingLookup(self.Q)(neg_item_id)
        # user_latent_factor = tf.nn.embedding_lookup(self.P, user_id)
        # item_latent_factor = tf.nn.embedding_lookup(self.Q, item_id)
        # neg_item_latent_factor = tf.nn.embedding_lookup(self.Q, neg_item_id)

        pred_y = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1)
        pred_y_neg = tf.reduce_sum(tf.multiply(user_latent_factor, neg_item_latent_factor), 1)
        self.model = Model(inputs=[self.user_id, self.item_id, self.neg_item_id],
                           outputs=[pred_y, pred_y_neg])
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):
        """
        You must prepare the data before train and test the model
        :param train_data:
        :param test_data:
        :return:
        """
        t = train_data.tocoo()
        self.user = t.row.reshape(-1)
        self.item = t.col.reshape(-1)
        self.num_training = len(self.item)
        self.test_data = test_data
        self.total_batch = int(self.num_training / self.batch_size)
        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")

    @tf.function
    def train_op(self, batch_user, batch_item, batch_item_neg):
        with tf.GradientTape() as tape:
            pred_y, pred_y_neg = self.model([batch_item, batch_user, batch_item_neg])
            loss = - tf.reduce_sum(
                tf.math.log(tf.sigmoid(pred_y - pred_y_neg))) + \
                   self.reg_rate * (tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q))
        gradients_of_model = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_of_model, self.model.trainable_variables))
        self.loss_estimator(loss)

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
            batch_user = np.array(user_random[i * self.batch_size:(i + 1) * self.batch_size])
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item_neg = item_random_neg[i * self.batch_size:(i + 1) * self.batch_size]

            np_batch_user = np.expand_dims(np.array(batch_user), -1)
            np_batch_item = np.expand_dims(np.array(batch_item), -1)
            np_batch_item_neg = np.expand_dims(np.array(batch_item_neg), -1)
            self.train_op(np_batch_user, np_batch_item, np_batch_item_neg)

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
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        user_id = tf.expand_dims(tf.convert_to_tensor(user_id), -1)
        item_id = tf.expand_dims(tf.convert_to_tensor(item_id), -1)
        dummy_neg_id = tf.zeros(item_id.shape, tf.int32)
        pred_y, pred_y_neg = self.model([user_id, item_id, dummy_neg_id])
        return pred_y.numpy()

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items
