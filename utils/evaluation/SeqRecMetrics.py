#!/usr/bin/env python
"""
Evaluation Metrics for Top N Recommendation
"""

import numpy as np
import time
from numpy.linalg import norm

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


import math

# efficient version
def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]

    count = len(hits)

    count_test = len(test_matrix)


    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return count, float(count / len(test_matrix)), float(dcg_k / idcg_k)

# def hitratio(k, rankedlist, test_matrix)

def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
    # for idx, vale in enumerate(b1):
    #     print(idx, vale)
    count = len(hits)


    for c in range(count):
        ap += (c+1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    max = len(b1) - 1
    if count != 0:
        count_test = max - hits[0][0]
    else:
        count_test = 0
    auc = 1.0 * count_test /max
    return auc, mrr, float(dcg / idcg)


def hitratio_at_k():
    print(" test")

def ndcg_at_k():
    print("test")


def evaluate1(self):
    pred_ratings_10 = {}
    pred_ratings_50 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    hr_at_50 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    auc = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    start_time = time.time()
    user_factors, seq_factors, user_factors_2, _ = self.getUserParam(self.test_users)
    item_factors_1, item_factors_2, bias_item = self.getItemParam(np.expand_dims(np.arange(self.num_item), axis=1))

    #
    print(np.shape(user_factors))
    # print(np.shape(seq_factors))
    print(np.shape(item_factors_1))
    # print(np.shape(item_factors_2))
    # print(type(user_factors))
    # print(type(seq_factors))
    # print(type(item_factors_1))
    # print(type(item_factors_2))

    # print(np.shape(user_factors[:,None]- item_factors_1))
    # print(np.shape(bias_item))
    results =  - self.alpha * np.sum((user_factors[:,None]- item_factors_1)**2, axis=2) \
               - (1-  self.alpha) *  np.sum((seq_factors[:,None]  - item_factors_2)**2, axis=2)
    #results =  -   np.sum((seq_factors[:,None] +  user_factors[:,None]- item_factors_2)**2, axis=2)

    #- np.reshape(bias_item, [ np.shape(bias_item)[1], np.shape(bias_item)[0]])
    # print(np.shape(results))
    # print(time.time() - start_time)
    for u in self.test_users:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        scores = []


        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)


            scores.append(results[u, j])


        #scores = self.predict(user_ids, item_ids)
        #print(np.shape(scores))
        #print( scores)
        #

        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)

        # print(ranked_list[u])
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_50[u] = pred_ratings[u][:50]


        hr, _, _ = precision_recall_ndcg_at_k(50, pred_ratings_50[u], self.test_data[u])
        # if hr > 0:
        #     print(u)
        #     print(self.test_sequences[u, :])
        #     print( self.test_data[u])
        #     print(seq_weights[u])
        auc_t, mrr_t, _ = map_mrr_ndcg(pred_ratings[u], self.test_data[u])

        hr_at_50.append(hr)
        mrr.append(mrr_t)
        auc.append(auc_t)
    print(np.sum(hr_at_50))
    print("------------------------")
    print("HR@50:" + str(np.mean(hr_at_50)))
    print("MRR:" + str(np.mean(mrr)))
    print("AUC:" + str(np.mean(auc)))

def evaluate_caser(self):
    pred_ratings_10 = {}
    pred_ratings_50 = {}
    pred_ratings = {}
    ranked_list = {}
    hr_at_50 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    all_users = np.arange(500)

    user_factors = self.getUserParam(self.test_users)
    item_factors, bias_item = self.getItemParam(np.expand_dims(np.arange(self.num_item), axis=1))

    #

    # print(np.shape(item_factors_1))
    # print(np.shape(item_factors_2))
    # print(type(user_factors))
    # print(type(seq_factors))
    # print(type(item_factors_1))
    # print(type(item_factors_2))

    # print(np.shape(user_factors[:,None]- item_factors_1))
    # print( bias_item[0] )
    #res = tf.reduce_sum(tf.multiply(x, self.w_items), 2) + self.b_items

    # print(np.shape(bias_item))
    # print(np.shape( np.dot(user_factors, item_factors.T)))
    results = np.dot(user_factors, item_factors.T) + bias_item

    # - self.alpha * np.sum((user_factors[:, None] - item_factors_1) ** 2, axis=2) \
    #       - (1 - self.alpha) * np.sum((seq_factors[:, None] - item_factors_2) ** 2, axis=2)

    for u in self.test_users:#all_users:#
        user_ids = []
        user_neg_items = self.neg_items[u] # self.all_items
        item_ids = []
        scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)
            scores.append(results[u, j])
        #scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)

        # print(ranked_list[u])
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_50[u] = pred_ratings[u][:50]

        hr, _, _ = precision_recall_ndcg_at_k(50, pred_ratings_50[u], self.test_data[u])
        _, mrr_t, _ = map_mrr_ndcg(pred_ratings[u], self.test_data[u])

        hr_at_50.append(hr)
        mrr.append(mrr_t)
    print(np.sum(hr_at_50))
    print("------------------------")
    print("HR@50:" + str(np.mean(hr_at_50)))
    print("MRR:" + str(np.mean(mrr)))

def evaluate(self):
    pred_ratings_10 = {}
    pred_ratings_50 = {}
    pred_ratings = {}
    ranked_list = {}
    hr_at_50 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    all_users = np.arange(500)
    for u in self.test_users:#all_users:#
        user_ids = []
        user_neg_items = self.neg_items[u] # self.all_items
        item_ids = []
        #scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)

        # print(ranked_list[u])
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_50[u] = pred_ratings[u][:50]

        hr, _, _ = precision_recall_ndcg_at_k(50, pred_ratings_50[u], self.test_data[u])
        _, mrr_t, _ = map_mrr_ndcg(pred_ratings[u], self.test_data[u])

        hr_at_50.append(hr)
        mrr.append(mrr_t)
    print(np.sum(hr_at_50))
    print("------------------------")
    print("HR@50:" + str(np.mean(hr_at_50)))
    print("MRR:" + str(np.mean(mrr)))
