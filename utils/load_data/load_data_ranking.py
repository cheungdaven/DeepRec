import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_data_all(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'time'],
                  test_size=0.2, sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_dict[(u, i)] = 1

    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                train_rating.append(1)
            else:
                train_rating.append(0)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))

    neg_items = {}
    train_interaction_matrix = []
    for u in range(n_users):
        neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    return train_interaction_matrix, test_dict, n_users, n_items


def load_data_neg(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                  test_size=0.2, sep="\t"):
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
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items


def load_data_separately(path_train=None, path_test=None, path_val=None, header=['user_id', 'item_id', 'rating'],
                         sep=" ", n_users=0, n_items=0):
    n_users = n_users
    n_items = n_items
    print("start")
    train_matrix = None
    if path_train is not None:
        train_data = pd.read_csv(path_train, sep=sep, names=header, engine='python')
        print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

        train_row = []
        train_col = []
        train_rating = []

        for line in train_data.itertuples():
            u = line[1]  # - 1
            i = line[2]  # - 1
            train_row.append(u)
            train_col.append(i)
            train_rating.append(1)

        train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    test_dict = None
    if path_test is not None:
        test_data = pd.read_csv(path_test, sep=sep, names=header, engine='python')
        test_row = []
        test_col = []
        test_rating = []
        for line in test_data.itertuples():
            test_row.append(line[1])
            i = line[2]  # - 1
            test_col.append(i)
            test_rating.append(1)

        test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

        test_dict = {}
        for u in range(n_users):
            test_dict[u] = test_matrix.getrow(u).nonzero()[1]
    all_items = set(np.arange(n_items))
    train_interaction_matrix = []
    for u in range(n_users):
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    if path_val is not None:
        val_data = pd.read_csv(path_val, sep=sep, names=header, engine='python')

    print("end")
    return train_interaction_matrix, test_dict, n_users, n_items
