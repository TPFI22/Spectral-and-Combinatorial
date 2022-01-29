import torch
from models import MyGNN
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToDevice
from transforms import SpectralTransform, ConstFeaturesTransform
from utils import separate_data, train, test
from torch import optim
from tqdm import tqdm
import os.path
from os import path
import shutil
import numpy as np
import argparse
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Ablation study')
parser.add_argument('--GNN', type=str, default="GIN",
                    help='GNN type (default: GIN)')
parser.add_argument('--dataset', type=str, default="NCI1",
                    help='Name of dataset (default: NCI1)')


args = parser.parse_args()
model_name,dataset_name =args.GNN, args.dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

t_ranges = [
    torch.logspace(-2, 2, steps=5),
    torch.logspace(-2, 2, steps=10),
    torch.logspace(-1, 1, steps=10),
    torch.logspace(-3, 3, steps=10),
    torch.logspace(-2, 2, steps=20),
    torch.logspace(-1, 1, steps=5),
    torch.logspace(-3, 3, steps=20),
]
with_max_options = [True, False, "median_and_min"]

for i, t_range in enumerate(t_ranges):
    for with_max in with_max_options:
        with_median_and_min = False
        if with_max == "median_and_min":
            if i < 2:
                with_max = True
                with_median_and_min = True
            else:
                continue
        if path.isdir(f"./dataset_spectral/{dataset_name}/"):
            shutil.rmtree(f"./dataset_spectral/{dataset_name}/")
        dataset = TUDataset(root=f'./dataset_spectral', name=dataset_name, use_node_attr=True,
                            transform=ToDevice(device),
                            pre_transform=SpectralTransform(t_range=t_range, with_max=with_max,
                                                            with_median_and_min=with_median_and_min))
        accs = np.zeros(10)
        for fold_idx in range(10):
            model = MyGNN(model_name, in_channels=dataset.num_node_features, hidden_channels=64,
                          final_dropout=0, num_classes=dataset.num_classes)
            model.to(device)
            train_graphs, test_graphs = separate_data(dataset, seed=0, fold_idx=fold_idx)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            for epoch in tqdm(range(0, 200)):
                train(256, model, train_graphs, optimizer)
                scheduler.step()
            accs[fold_idx] = test(model, test_graphs)
        f = open(f"ablation_{dataset_name}_{model_name}.txt", "a")
        f.write(
            f'range: {i} with_max: {with_max} with_median_and_min: {with_median_and_min}, has {accs.mean()}±{accs.std()} test accuracy')
        f.close()

print(f"Full ablation study results for {dataset_name} and {model_name} is at 'ablation_{dataset_name}_{model_name}.txt'")
