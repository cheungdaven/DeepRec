import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def load_data_rating(path="B:/Datasets/MovieLens/ml-100k/u.data", header = ['user_id', 'item_id', 'rating', 'category'], test_size = 0.1, sep="\t"):

    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items

# def load_data_autorec(path="B:/Datasets/MovieLens/ml-100k/u.data", header = ['user_id', 'item_id', 'rating', 'category'], test_size = 0.1, sep="\t"):
#     fp = open("./Data/movielens_100k.dat")
#
#     df = pd.read_csv(path, sep=sep, names=header, engine='python')
#
#     n_users = df.user_id.unique().shape[0]
#     n_items = df.item_id.unique().shape[0]
#
#     print(n_users)
#     print(n_items)
#
#     user_train_set = set()
#     user_test_set = set()
#     item_train_set = set()
#     item_test_set = set()
#
#     train_R = np.zeros((n_users, n_items))
#     test_R = np.zeros((n_users, n_items))
#
#     random_perm_idx = np.random.permutation(num_total_ratings)
#     train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]
#     test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]
#
#     num_train_ratings = len(train_idx)
#     num_test_ratings = len(test_idx)
#
#     lines = fp.readlines()
#
#     ''' Train '''
#     for itr in train_idx:
#         line = lines[itr]
#         user, item, rating, _ = line.split("\t")
#         user_idx = int(user) - 1
#         item_idx = int(item) - 1
#         train_R[user_idx, item_idx] = int(rating)
#         user_train_set.add(user_idx)
#         item_train_set.add(item_idx)
#
#
#     ''' Test '''
#     for itr in test_idx:
#         line = lines[itr]
#         user, item, rating, _ = line.split("\t")
#         user_idx = int(user) - 1
#         item_idx = int(item) - 1
#         test_R[user_idx, item_idx] = int(rating)
#         user_test_set.add(user_idx)
#         item_test_set.add(item_idx)
#
#
#
#     return train_R, test_R, num_train_ratings, num_test_ratings, \
#            user_train_set, item_train_set, user_test_set, item_test_set