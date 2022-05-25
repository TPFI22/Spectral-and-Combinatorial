import torch;

torch.manual_seed(0)
from torch import optim
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToDevice,LocalDegreeProfile,Compose
from utils import *
from transforms import *
from models import *
import pickle
from tqdm import tqdm

datasets = ['MUTAG', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'PROTEINS', 'PTC_MR', 'REDDIT-BINARY',
            'REDDIT-MULTI-5K']


def generate_classification_results(model_name, dataset_name, hidden_layers_list, batch_size=128, epochs=700,
                                    final_dropouts=[0, 0.5], features='standard', with_max=False, lr=0.01, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    t_range = torch.logspace(-2, 2, steps=5) if with_max else torch.logspace(-2, 2, steps=10)
    if features == 'spectral':
        dataset = TUDataset(root=f'./dataset_spectral', name=dataset_name, use_node_attr=True,
                            transform=ToDevice(device),
                            pre_transform=SpectralTransform(t_range=t_range))
    elif features == 'standard':
        dataset = TUDataset(root=f'./dataset_new', name=dataset_name, use_node_attr=True,
                            transform=ToDevice(device),
                            pre_transform=ConstFeaturesTransform())
    elif features == 'subgraph':
        dataset = TUDataset(root=f'./dataset_subgraph', name=dataset_name, use_node_attr=True,
                            #transform=ToDevice(device),
                            pre_transform=Compose([ThreeSubgraphIsomorphismTransform(),MinMaxNormalizationTransform(),ToDevice(device)]))
    elif features == 'degree':
        dataset = TUDataset(root=f'./dataset_degree', name=dataset_name, use_node_attr=True,
                            #transform=ToDevice(device),
                            pre_transform=Compose([LocalDegreeProfile(),MinMaxNormalizationTransform(),ToDevice(device)]))
    for final_dropout in final_dropouts:
        for hidden_layers in hidden_layers_list:
            for fold in list(range(10)):
                model = MyGNN(model_name, in_channels=dataset.num_node_features, hidden_channels=hidden_layers,
                              final_dropout=final_dropout, num_classes=dataset.num_classes)
                model.to(device)
                train_graphs, test_graphs = separate_data(dataset, seed=seed, fold_idx=fold)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                test_accs = []
                for epoch in tqdm(range(0, epochs)):
                    train(batch_size, model, train_graphs, optimizer)
                    acc_test = test(model, test_graphs)
                    test_accs.append(acc_test)
                    scheduler.step()
                with open(
                        f"./results/{model_name}/{dataset_name}/FD={str(final_dropout)}-HL={hidden_layers}-F={features}-FOLD={fold}",
                        "wb") as fp:
                    pickle.dump(test_accs, fp)
