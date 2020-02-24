import argparse
import tensorflow as tf
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.item_ranking.cdae import ICDAE
from models.item_ranking.bprmf import BPRMF
from models.item_ranking.cml import CML
from models.item_ranking.neumf import NeuMF
from models.item_ranking.gmf import GMF
from models.item_ranking.jrl import JRL
from models.item_ranking.mlp import MLP
from models.item_ranking.lrml import LRML

from utils.load_data.load_data_ranking import *


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['CDAE', 'CML', 'NeuMF', 'GMF', 'MLP', 'BPRMF', 'JRL', 'LRML'],
                        default='LRML')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)  # 128 for unlimpair
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

    train_data, test_data, n_user, n_item = load_data_neg(test_size=0.2, sep="\t")

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    model = None
    # Model selection
    if args.model == "CDAE":
        train_data, test_data, n_user, n_item = load_data_all(test_size=0.2, sep="\t")
        model = ICDAE(n_user, n_item)
    if args.model == "CML":
        model = CML(n_user, n_item)
    if args.model == "LRML":
        model = LRML(n_user, n_item)
    if args.model == "BPRMF":
        model = BPRMF(n_user, n_item)
    if args.model == "NeuMF":
        model = NeuMF(n_user, n_item)
    if args.model == "GMF":
        model = GMF(n_user, n_item)
    if args.model == "MLP":
        model = MLP(n_user, n_item)
    if args.model == "JRL":
        model = JRL(n_user, n_item)
    # build and execute the model
    if model is not None:
        model.build_network()
        model.execute(train_data, test_data)
