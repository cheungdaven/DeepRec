import argparse
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.seq_rec.Caser import Caser
from models.seq_rec.AttRec import AttRec
from models.seq_rec.PRME import PRME
from utils.load_data.load_data_seq import DataSet
from utils.load_data.load_data_ranking import *


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['Caser','PRME', 'AttRec'], default = 'AttRec')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=256)
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


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True



    with tf.Session(config=config) as sess:
        model = None
        # Model selection

        if args.model == "Caser":
            train_data = DataSet(path="../data/ml100k/temp/train.dat", sep="\t",
                                 header=['user', 'item', 'rating', 'time'],
                                 isTrain=True, seq_len=5, target_len=3, num_users=943, num_items=1682)
            test_data = DataSet(path="../data/ml100k/temp/test.dat", sep="\t",
                                header=['user', 'item', 'rating', 'time'],
                                user_map=train_data.user_map, item_map=train_data.item_map)
            # train_data = DataSet(path="../Data/ml100k/seq/train.dat",  isTrain=True)
            # test_data = DataSet(path="../Data/ml100k/seq/test.dat", user_map=train_data.user_map, item_map=train_data.item_map)
            model = Caser(sess, train_data.num_user,  train_data.num_item)
            model.build_network(L = train_data.sequences.L, num_T=train_data.sequences.T)
            model.execute(train_data, test_data)
        if args.model == "PRME":
            train_data = DataSet(path="../data/ml100k/temp/train.dat", sep="\t",header=['user', 'item', 'rating', 'time'],isTrain=True, seq_len=1, target_len=1)
            test_data = DataSet(path="../data/ml100k/temp/test.dat", sep="\t", header=['user', 'item', 'rating', 'time'], user_map=train_data.user_map, item_map=train_data.item_map)
            model = PRME(sess, train_data.num_user,  train_data.num_item)
            model.build_network(L = train_data.sequences.L, num_T=train_data.sequences.T)
            model.execute(train_data, test_data)
        if args.model == "AttRec":
            train_data = DataSet(path="../data/ml100k/temp/train.dat", sep="\t",header=['user', 'item', 'rating', 'time'],isTrain=True, seq_len=5, target_len=3, num_users=943, num_items=1682)
            test_data = DataSet(path="../data/ml100k/temp/test.dat", sep="\t", header=['user', 'item', 'rating', 'time'], user_map=train_data.user_map, item_map=train_data.item_map)
            model = AttRec(sess, train_data.num_user,  train_data.num_item)
            # print(train_data.user_map)
            # print(train_data.item_map)
            model.build_network(L = train_data.sequences.L, num_T=train_data.sequences.T)
            model.execute(train_data, test_data)
