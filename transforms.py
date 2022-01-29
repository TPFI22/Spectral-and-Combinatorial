import torch;

torch.manual_seed(0)
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import Union
from torch_geometric.utils import to_undirected
import networkx as nx
from networkx import laplacian_matrix


class ConstFeaturesTransform(BaseTransform):
    r"""Adds a constant value of 1 to each node feature  without features :obj:`x`. """

    def __init__(self):
        BaseTransform.__init__(self)

    def __call__(self, data: Union[Data, HeteroData]):

        for store in data.node_stores:
            c = torch.full((store.num_nodes, 1), 1, dtype=torch.float)
            if not hasattr(store, 'x'):
                store.x = c
        return data


class SpectralTransform(BaseTransform):
    r"""Adds a spectral values to each node feature :obj:`x`."""

    def __init__(self, t_range=torch.logspace(-2, 2, steps=10), with_max=False, with_median_and_min=False):
        BaseTransform.__init__(self)
        self.t_range = t_range
        self.with_max = with_max
        self.with_median_and_min = with_median_and_min

    @staticmethod
    def __heat_kernel_features__(edge_index: torch.Tensor, num_nodes, t_range=torch.logspace(-2, 2, steps=10),
                                 with_max=False, with_median_and_min=False):
        g = nx.Graph()
        g.add_nodes_from(list(range(num_nodes)))
        g.add_edges_from((to_undirected(edge_index).T.tolist()))
        adj_mat = torch.tensor(nx.to_numpy_array(g))
        n = len(adj_mat)
        T = len(t_range)
        t_range = t_range.double()  # .to(device)
        laplacian = torch.Tensor(laplacian_matrix(nx.from_numpy_array(adj_mat.numpy())).toarray())  # .to(device)
        eig_val, eig_vec = torch.linalg.eigh(laplacian.double())
        features = None
        if n < 500:
            eigen_vec_3d = eig_vec.unsqueeze(0).repeat(n, 1, 1) * (eig_vec.unsqueeze(1).repeat(1, n, 1))
            H = torch.exp(-eig_val.unsqueeze(1) @ t_range.unsqueeze(0)).T @ eigen_vec_3d.transpose(1, 2)
            H = H.transpose(1, 2).transpose(0, 1)
            features = H[range(len(H)), range(len(H))].clone()
        elif n < 3500:
            H = torch.zeros(n, n, T)  # .to(device)
            batch_size = 5
            for j in range(0, n, batch_size):
                start = j
                end = min(j + batch_size, n)
                eigen_vec_3d = (eig_vec[start:end] * (eig_vec.unsqueeze(1).repeat(1, end - start, 1))).transpose(1, 2)
                H[start:end, :, :] = (
                        torch.exp(-eig_val.unsqueeze(1) @ t_range.unsqueeze(0)).T @ eigen_vec_3d).transpose(1,
                                                                                                            2).transpose(
                    0, 1)
            features = H[range(len(H)), range(len(H))].clone()
        else:
            lst = []
            eig_vec2 = torch.pow(eig_vec, 2)
            stamps = torch.exp((t_range.repeat((eig_val.size()[0], 1)).T * - eig_val).T)
            for i in range(num_nodes):
                lst.append((stamps.T * eig_vec2[i, :]).T.sum(0))
            features = torch.stack(lst)
        if with_max:
            max_features = []
            for i in range(H.size()[2]):
                H[:, :, i] = H[:, :, i] - torch.diag(torch.ones(H.size()[0]))
                max_features.append(torch.max(H[:, :, i], dim=1)[0])
                H[:, :, i] = H[:, :, i] + torch.diag(torch.ones(H.size()[0]))
            features = torch.concat((features, torch.stack(max_features).T), dim=1)
        if with_median_and_min:
            min_features = []
            median_features = []
            for i in range(H.size()[2]):
                H[:, :, i] = H[:, :, i] + torch.diag(torch.ones(H.size()[0]))
                min_features.append(torch.max(H[:, :, i], dim=1)[0])
                median_features.append(torch.median(H[:, :, i], dim=1)[0])
                H[:, :, i] = H[:, :, i] - torch.diag(torch.ones(H.size()[0]))
            features = torch.concat((features, torch.stack(min_features).T), dim=1)
            features = torch.concat((features, torch.stack(median_features).T), dim=1)
        return features.float()

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            c = self.__heat_kernel_features__(store.edge_index, store.num_nodes, t_range=self.t_range,
                                              with_max=self.with_max, with_median_and_min=self.with_median_and_min)

            if hasattr(store, 'x'):
                x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
            else:
                store.x = c
        return data
