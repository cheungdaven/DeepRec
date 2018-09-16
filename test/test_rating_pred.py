import argparse
import tensorflow as tf
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.rating_prediction.nnmf import NNMF
from models.rating_prediction.mf import MF
from models.rating_prediction.nrr import NRR
from models.rating_prediction.autorec import *
from models.rating_prediction.nfm import NFM
from utils.load_data.load_data_rating import *
from utils.load_data.load_data_content import *

def parse_args():
    parser = argparse.ArgumentParser(description='nnRec')
    parser.add_argument('--model', choices=['MF', 'NNMF', 'NRR', 'I-AutoRec', 'U-AutoRec', 'FM', 'NFM', 'AFM'], default='U-AutoRec')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)  # 128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1)  # 0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size

    train_data, test_data, n_user, n_item = load_data_rating(path="../Data/ml100k/movielens_100k.dat",
                                                             header=['user_id', 'item_id', 'rating', 't'],
                                                             test_size=0.1, sep="\t")


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = None
        # Model selection
        if args.model == "MF":
            model = MF(sess, n_user, n_item, batch_size=batch_size)
        if args.model == "NNMF":
            model = NNMF(sess, n_user, n_item, learning_rate=learning_rate)
        if args.model == "NRR":
            model = NRR(sess, n_user, n_item)
        if args.model == "I-AutoRec":
            model = IAutoRec(sess, n_user, n_item)
        if args.model == "U-AutoRec":
            model = UAutoRec(sess, n_user, n_item)
        if args.model == "NFM":
            train_data, test_data, feature_M = load_data_fm()
            n_user = 957
            n_item = 4082
            model = NFM(sess, n_user, n_item)
            model.build_network(feature_M)
            model.execute(train_data, test_data)
        # build and execute the model
        if model is not None:
            model.build_network()
            model.execute(train_data, test_data)
