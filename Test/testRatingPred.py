import argparse
import tensorflow as tf

from RatingPrediction.NNMF import NNMF
from RatingPrediction.MF import MF
from RatingPrediction.NRR import NRR

from LoadData.load_data_rating import *

def parse_args():
    parser = argparse.ArgumentParser(description='nnRec')
    parser.add_argument('--model', choices=['MF','NNMF','NRR', 'combined', 'unlimpair', 'nlinearpair','neurecplus'], default = 'NRR')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024 ) #128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3) #1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1) #0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size



    train_data, test_data, n_user, n_item  = load_data( test_size=0.1, sep="\t")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    if args.model == "MF":
        with tf.Session(config=config) as sess:
            model = MF(sess, n_user, n_item)
            model.initialize()
            model.fit(train_data, test_data)
    if args.model == "NNMF":
        with tf.Session(config=config) as sess:
            model = NNMF(sess, n_user, n_item, learning_rate=learning_rate)
            model.initialize()
            model.fit(train_data, test_data)
    if args.model == "NRR":
        with tf.Session(config=config) as sess:
            model = NRR(sess, n_user, n_item)
            model.initialize()
            model.fit(train_data, test_data)
            model.predict([1,2],[1,8])