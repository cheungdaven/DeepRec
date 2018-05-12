import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def load_data(path="B:/Datasets/MovieLens/ml-100k/u.data", header = ['user_id', 'item_id', 'rating', 'category'], test_size = 0.1, sep="\t"):

    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print(n_users)
    print(n_items)



    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []
    train_rating_1= []

    train_dict = {}
    for line in  df.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        r = line[3]
        if (u,i) in test_data:
            continue
        train_dict[(u, i)] = r

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
        train_rating_1.append(1)


    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))


    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))


    return train_matrix.todok(), test_matrix.todok(), n_users, n_items
