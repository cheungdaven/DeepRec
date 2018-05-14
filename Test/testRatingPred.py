import argparse
import tensorflow as tf

from RatingPrediction.NNMF import NNMF
from RatingPrediction.MF import MF
from RatingPrediction.NRR import NRR
from RatingPrediction.AutoRec import *

from LoadData.load_data_rating import *

def parse_args():
    parser = argparse.ArgumentParser(description='nnRec')
    parser.add_argument('--model', choices=['MF','NNMF','NRR', 'I-AutoRec', 'U-AutoRec'], default = 'I-AutoRec')
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

    train_data, test_data, n_user, n_item = load_data_rating(test_size=0.1, sep="\t")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = None
        # Model selection
        if args.model == "MF":
            model = MF(sess, n_user, n_item)
        if args.model == "NNMF":
            model = NNMF(sess, n_user, n_item, learning_rate=learning_rate)
        if args.model == "NRR":
            model = NRR(sess, n_user, n_item)
        if args.model == "I-AutoRec":
            model = IAutoRec(sess, n_user, n_item)
        if args.model == "U-AutoRec":
            model = UAutoRec(sess, n_user, n_item)

        # build and execute the model
        if model is not None:
            model.build_network()
            model.execute(train_data, test_data)