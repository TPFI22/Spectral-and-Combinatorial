import numpy as np
import torch
from transforms import SpectralTransform
from utils import train, test
from tqdm import tqdm
import networkx as nx
import random
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from torch import optim
from models import MyGNN

wl1_fail = (
    np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0], 
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
    ]),
    np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],  
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0], 
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
    ]))

wl2_fail = (np.array([
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
]), np.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
]))

spectral_fail = (np.array([
    [0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
]),
                 np.array([
                     [0, 1, 0, 0, 0, 1, 1, 0],
                     [1, 0, 0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0, 1, 0, 0],
                     [1, 0, 0, 1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 0, 0],
                 ]))


def array2data(array: np.array, x: torch.Tensor, y):
    g = nx.from_numpy_array(array)
    edge_index = torch.tensor(list(g.edges())).T
    if edge_index.size()[0] != 0:
        edge_index = to_undirected(edge_index)
    else:
        edge_index = torch.Tensor([[], []]).long()
    data = Data(x, edge_index, y=y)
    return data


def random_permutation(n: int):
    p = np.random.permutation(np.eye(n))
    return p


def separate_data(dataset, seed, fold_idx):
    assert 0 <= fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    idx_list = []
    for idx in skf.split(np.zeros(len(dataset)), np.zeros(len(dataset))):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return list(dataset[i] for i in list(train_idx)), list(dataset[i] for i in list(test_idx))


def generate_dataset(graphs, num=1000):
    a1, a2 = graphs
    nodes = a1.shape[0]
    all_possible_edges = []
    for i in range(nodes - 1):
        for j in range(i + 1, nodes):
            all_possible_edges += [(i, j)]

    g1 = nx.from_numpy_array(a1)
    g2 = nx.from_numpy_array(a2)
    all_edges_to_add1 = list(set(all_possible_edges) - set(g1.edges()))
    all_edges_to_add2 = list(set(all_possible_edges) - set(g2.edges()))
    all_edges_to_remove1 = list(g1.edges())
    all_edges_to_remove2 = list(g2.edges())

    pertubated_graphs = []
    for _ in range(num // 2):
        pertub = random.randint(-1, 2)
        g1 = nx.from_numpy_array(a1)
        g2 = nx.from_numpy_array(a2)
        if pertub == -1:
            remove1 = random.choice(all_edges_to_remove1)
            remove2 = random.choice(all_edges_to_remove2)
            g1.remove_edge(remove1[0], remove1[1])
            g2.remove_edge(remove2[0], remove2[1])
        elif pertub == 1:
            add1 = random.choice(all_edges_to_add1)
            add2 = random.choice(all_edges_to_add2)
            g1.add_edge(add1[0], add1[1])
            g2.add_edge(add2[0], add2[1])

        b1 = nx.to_numpy_array(g1)
        b2 = nx.to_numpy_array(g2)

        p = random_permutation(nodes)
        pertubated_graphs += [array2data(p @ b1 @ p.T, x=torch.ones((nodes, 1)), y=0)]

        p = random_permutation(nodes)
        pertubated_graphs += [array2data(p @ b2 @ p.T, x=torch.ones((nodes, 1)), y=1)]
    return pertubated_graphs


def add_spectral_features(dataset):
    for i in range(len(dataset)):
        dataset[i].x = SpectralTransform.__heat_kernel_features__(dataset[i].edge_index, dataset[i].num_nodes,
                                                                  t_range=torch.logspace(-2, 2, steps=10))



																  
repeats = 100
graph_couples = [wl1_fail, spectral_fail]
accs_spectral = [np.zeros(repeats), np.zeros(repeats), np.zeros(repeats)]
accs_standard = [np.zeros(repeats), np.zeros(repeats), np.zeros(repeats)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for r in range(repeats):
    for j, original_graphs in enumerate(graph_couples):
        dataset = generate_dataset(original_graphs)
        for i in range(len(dataset)):
            dataset[i] = dataset[i].to(device)
        train_graphs, test_graphs = separate_data(dataset, seed=0, fold_idx=0)

        vanilla_model = MyGNN("GAT", in_channels=1, hidden_channels=64,
                              final_dropout=0, num_classes=2).to(device)
        optimizer = optim.Adam(vanilla_model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in tqdm(range(0, 100)):
            train(256, vanilla_model, train_graphs, optimizer)
            scheduler.step()

        accs_standard[j][r] = test(vanilla_model, test_graphs)

        add_spectral_features(train_graphs)
        add_spectral_features(test_graphs)
        for i in range(len(dataset)):
            dataset[i] = dataset[i].to(device)
        sepctral_model = MyGNN("GAT", in_channels=20, hidden_channels=64,
                               final_dropout=0, num_classes=2).to(device)
        optimizer = optim.Adam(sepctral_model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        for epoch in tqdm(range(0, 100)):
            train(256, sepctral_model, train_graphs, optimizer)
            scheduler.step()

        accs_spectral[j][r] = test(sepctral_model, test_graphs)

print(
    f"1-WL fail results: SP-GAT:{accs_spectral[0].mean()}±{accs_spectral[0].std()}, GAT: {accs_standard[0].mean()}±{accs_standard[0].std()}")
print(
    f"spectral fail results: SP-GAT:{accs_spectral[1].mean()}±{accs_spectral[1].std()}, GAT: {accs_standard[1].mean()}±{accs_standard[1].std()}")