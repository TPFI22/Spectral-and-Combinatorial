import argparse
from typing import List
from find_best_results import print_classification_results
from generate_classification_results import generate_classification_results

parser = argparse.ArgumentParser(description='PyTorch Geometric graph neural net for whole-graph classification')
parser.add_argument('--GNN', type=str, default="GIN",
                    help='GNN type (default: GIN)')
parser.add_argument('--dataset', type=str, default="MUTAG",
                    help='name of dataset (default: MUTAG)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for splitting the dataset into 10 folds(default: 0)')
parser.add_argument('--num_layers', type=int, default=5,
                    help='number of message passing layers (default: 5)')
parser.add_argument('--hidden_layers_list', metavar='N', type=int, nargs='+',
                    help='List of hidden layers to test', default=[64])
parser.add_argument('--features', type=str, default="standard",
                    help='The graph features used (default: standard)')
parser.add_argument('--with_max', type=bool, default=False,
                    help='Use the maximum quantile (default: False')
parser.add_argument('--final_dropouts', type=List, default=[0, 0.5],
                    help='final layer dropout (default: [0,0.5])')

args = parser.parse_args()
generate_classification_results(args.GNN, args.dataset, args.hidden_layers_list, args.batch_size, args.epochs,
                                args.final_dropouts, args.features, args.with_max, args.lr, args.seed)

print_classification_results(args.GNN, args.dataset, args.hidden_layers_list, args.epochs,
                             args.final_dropouts, args.features)
