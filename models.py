import torch;

from torch import Tensor
from torch.nn.functional import dropout
from torch_geometric.nn import BatchNorm, GIN, GCN, GraphSAGE, GAT
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_add
torch.manual_seed(0)


class MyGNN(torch.nn.Module):
    def __init__(self, name='GIN', in_channels=1, hidden_channels=64, num_layers=5, final_dropout=0.5, num_classes=2):
        super().__init__()
        name2constructor = {'GIN': GIN, 'GCN': GCN, 'GraphSAGE': GraphSAGE, 'GAT': GAT}
        constructor = name2constructor[name]
        self.gin_model = constructor(in_channels, hidden_channels, num_layers, norm=BatchNorm(hidden_channels),
                                     jk='cat')
        self.linear = torch.nn.Linear(in_channels + hidden_channels * num_layers, num_classes)
        self.final_dropout = final_dropout

    def forward(self, data: Batch) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch.to(data.x.device)
        node_desciptors = self.gin_model(x, edge_index)
        graph_descriptors = scatter_add(node_desciptors, batch, dim=0)
        first_graph_descriptor = scatter_add(x, batch, dim=0)
        graph_descriptors = torch.cat((first_graph_descriptor, graph_descriptors), 1)
        return dropout(self.linear(graph_descriptors), self.final_dropout, training=self.training)


class MyNodeGNN(torch.nn.Module):
    def __init__(self, name='GIN', in_channels=1, hidden_channels=32, num_layers=4, num_classes=2):
        super().__init__()
        name2constructor = {'GIN': GIN, 'GCN': GCN, 'GraphSAGE': GraphSAGE, 'GAT': GAT}
        constructor = name2constructor[name]
        self.gin_model = constructor(in_channels, hidden_channels, num_layers,
                                     norm=BatchNorm(hidden_channels), jk='cat')
        self.linear = torch.nn.Linear(in_channels + hidden_channels * num_layers, num_classes)

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        node_desciptors = torch.cat([x, self.gin_model(x, edge_index)], dim=1)
        pred = self.linear(node_desciptors)
        return pred
