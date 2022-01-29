import pickle
import random
import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.transforms import ToDevice
from torch_geometric import seed_everything
from utils import *
from models import MyNodeGNN
from transforms import SpectralTransform, ConstFeaturesTransform
from torch import optim
from tqdm import tqdm
import argparse

import numpy as np

seed_everything(0)


def best_node_classification_params(model_name, dataset_name, hidden_layers_list=[128, 256, 384, 512],
                                    epochs=200, num_layers_list=[2, 3, 4],
                                    features='standard', with_max=False, lr=0.01, repeats=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(dataset_name)
    print(model_name)
    t_range = torch.logspace(-2, 2, steps=5) if with_max else torch.logspace(-2, 2, steps=10)
    if features == 'spectral':
        if dataset_name == 'PPI':
            dataset = [PPI(root=f'./dataset_spectral',
                           split="train",
                           transform=ToDevice(device),
                           pre_transform=SpectralTransform()),
                       PPI(root=f'./dataset_spectral',
                           split="val",
                           transform=ToDevice(device),
                           pre_transform=SpectralTransform()),
                       PPI(root=f'./dataset_spectral',
                           split="test",
                           transform=ToDevice(device),
                           pre_transform=SpectralTransform())]
        else:
            dataset = Planetoid(root=f'./dataset_spectral', name=dataset_name,
                                transform=ToDevice(device),
                                pre_transform=SpectralTransform(t_range=t_range, with_max=with_max))
        print("spectral")
    else:
        if dataset_name == 'PPI':
            dataset = [PPI(root=f'./dataset_new',
                           split="train",
                           transform=ToDevice(device),
                           pre_transform=ConstFeaturesTransform()),
                       PPI(root=f'./dataset_new',
                           split="val",
                           transform=ToDevice(device),
                           pre_transform=ConstFeaturesTransform()),
                       PPI(root=f'./dataset_new',
                           split="test",
                           transform=ToDevice(device),
                           pre_transform=ConstFeaturesTransform())
                       ]

        else:
            dataset = Planetoid(root=f'./dataset_new', name=dataset_name,
                                transform=ToDevice(device),
                                pre_transform=ConstFeaturesTransform())
        print("standard")

    best_val_acc = -1
    best_val_std = 0
    best_params = None
    for hidden_layer in hidden_layers_list:
        for num_layers in num_layers_list:
            val_accs = np.zeros(epochs)
            val_stds = []
            for i in tqdm(range(repeats)):
                val_stds.append([])
                if dataset_name == "PPI":
                    model = MyNodeGNN(model_name, in_channels=dataset[0].num_node_features,
                                      hidden_channels=hidden_layer,
                                      num_layers=num_layers,
                                      num_classes=dataset[0].num_classes * 2)
                else:
                    model = MyNodeGNN(model_name, in_channels=dataset.num_node_features, hidden_channels=hidden_layer,
                                      num_layers=num_layers,
                                      num_classes=dataset.num_classes)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                for epoch in range(0, epochs):
                    if dataset_name == "PPI":
                        train_nodes_PPI(model, dataset[0], optimizer)
                    else:
                        train_nodes(model, dataset[0], optimizer)
                    scheduler.step()
                    if dataset_name != "PPI":
                        acc = validate_nodes(model, dataset[0])
                        val_accs[epoch] += acc
                        val_stds[i].append(acc)
                if dataset_name == "PPI":
                    acc = validate_nodes_PPI(model, dataset[1])
                    val_accs[epochs - 1] += acc
                    for _ in range(epochs - 1):
                        val_stds[i].append(0)
                    val_stds[i].append(acc)

            val_accs /= repeats
            val_acc, epoch = val_accs.max(), val_accs.argmax()
            val_std = np.array(val_stds)[:, epoch].std()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {"epoch": epoch, "hidden_layer": hidden_layer, "num_layers": num_layers}
                best_val_std = val_std

    test_accs = []
    for _ in range(repeats):
        if dataset_name == "PPI":
            model = MyNodeGNN(model_name, in_channels=dataset[0].num_node_features,
                              hidden_channels=best_params["hidden_layer"],
                              num_layers=best_params["num_layers"],
                              num_classes=dataset[0].num_classes * 2)
        else:
            model = MyNodeGNN(model_name, in_channels=dataset.num_node_features,
                              hidden_channels=best_params["hidden_layer"],
                              num_layers=best_params["num_layers"],
                              num_classes=dataset.num_classes)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        for _ in range(0, best_params["epoch"]):
            if dataset_name == "PPI":
                train_nodes_PPI(model, dataset[0], optimizer, dataset[1])
            else:
                train_nodes(model, dataset[0], optimizer, with_val=True)
            scheduler.step()
        if dataset_name == "PPI":
            test_accs += [test_nodes_PPI(model, dataset[2])]
        else:
            test_accs += [test_nodes(model, dataset[0])]
    test_acc = np.array(test_accs).mean()
    test_std = np.array(test_accs).std()

    print("best validation accuracy: ", best_val_acc, best_val_std)
    print("best hyperparameters: ", best_params)
    print("test accuracy: ", test_acc, test_std)


parser = argparse.ArgumentParser(description='Node classification experiment')
parser.add_argument('--GNN', type=str, default="GIN",
                    help='GNN type (default: GIN)')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='name of dataset (default: Cora)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--features', type=str, default="standard",
                    help='The graph features used (default: standard)')
parser.add_argument('--with_max', type=bool, default=False,
                    help='Use the maximum quantile (default: False')

args = parser.parse_args()

best_node_classification_params(args.GNN, args.dataset, args.features, args.with_max, args.epochs)
