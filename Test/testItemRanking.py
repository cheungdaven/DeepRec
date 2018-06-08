import argparse
import tensorflow as tf

from Models.ItemRanking.CDAE import ICDAE
from Models.ItemRanking.BPRMF import BPRMF
from Models.ItemRanking.CML import CML
from Models.ItemRanking.NeuMF import NeuMF
from Models.ItemRanking.GMF import GMF
from Models.ItemRanking.JRL import JRL
from Models.ItemRanking.MLP import MLP

from Utils.LoadData.load_data_ranking import *

def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['CDAE','CML','NeuMF', 'GMF', 'MLP', 'BPRMF', 'JRL'], default = 'MLP')
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

    train_data, test_data, n_user, n_item = load_data_neg(test_size=0.2, sep="\t")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = None
        # Model selection
        if args.model == "CDAE":
            train_data, test_data, n_user, n_item = load_data_all(test_size=0.2, sep="\t")
            model = ICDAE(sess, n_user, n_item)
        if args.model == "CML":
            model = CML(sess, n_user, n_item)
        if args.model == "BPRMF":
            model = BPRMF(sess, n_user, n_item)
        if args.model == "NeuMF":
            model = NeuMF(sess, n_user, n_item)
        if args.model == "GMF":
            model = GMF(sess, n_user, n_item)
        if args.model == "MLP":
            model = MLP(sess, n_user, n_item)
        if args.model == "JRL":
            model = JRL(sess, n_user, n_item)
        # build and execute the model
        if model is not None:
            model.build_network()
            model.execute(train_data, test_data)
