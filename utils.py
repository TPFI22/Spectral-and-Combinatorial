import torch;

torch.manual_seed(0)
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold


def pass_data_iteratively(model, graphs, batch_size=256):
    model.eval()
    output = []
    labels = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    for data in loader:
        output.append(model(data).detach())
        labels.append(data.y)
    return torch.cat(output, 0), torch.cat(labels, 0)


def test(model, test_graphs):
    model.eval()

    output, labels = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(labels.view_as(pred).to(output.device)).sum().item()
    acc_test = correct / float(len(test_graphs))

    return acc_test


def test_nodes(model, graph):
    model.eval()

    outputs = model(graph)
    pred = outputs.max(1, keepdim=True)[1].squeeze()
    correct = pred[graph.test_mask] == graph.y[graph.test_mask]  # Check against ground-truth labels.
    test_acc = int(correct.sum()) / int(graph.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def test_nodes_PPI(model, graphs):
    model.eval()

    sum_correct_preds = 0
    total_nodes = 0
    num_classes = graphs[0].y.size()[1]
    for g in graphs:
        outputs = model(g)
        for c in range(g.y.size()[1]):
            pred = outputs[:, [2 * c, 2 * c + 1]].max(1, keepdim=True)[1].squeeze()
            sum_correct_preds += (pred == g.y[:, c]).sum()  # Check against ground-truth labels.
        total_nodes += g.num_nodes * num_classes

    test_acc = sum_correct_preds / total_nodes  # Derive ratio of correct predictions.
    return test_acc.item()


def validate_nodes(model, graph):
    model.eval()

    outputs = model(graph)
    pred = outputs.max(1, keepdim=True)[1].squeeze()
    correct = pred[graph.val_mask] == graph.y[graph.val_mask]  # Check against ground-truth labels.
    val_acc = int(correct.sum()) / int(graph.val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc


def validate_nodes_PPI(model, graphs):
    model.eval()

    sum_correct_preds = 0
    total_nodes = 0
    num_classes = graphs[0].y.size()[1]
    for g in graphs:
        outputs = model(g)
        for c in range(g.y.size()[1]):
            pred = outputs[:, [2 * c, 2 * c + 1]].max(1, keepdim=True)[1].squeeze()
            sum_correct_preds += (pred == g.y[:, c]).sum()  # Check against ground-truth labels.
        total_nodes += g.num_nodes * num_classes

    test_acc = sum_correct_preds / total_nodes  # Derive ratio of correct predictions.
    return test_acc.item()


def train(batch_size, model, train_graphs, optimizer):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    loss_accum = 0
    total_iters = 0
    train_graphs_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    for batch_graphs in train_graphs_loader:
        total_iters += 1
        output = model(batch_graphs)
        labels = batch_graphs.y.to(batch_graphs.x.device)
        # compute loss
        loss = criterion(output, labels)
        loss_accum += loss

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    average_loss = loss_accum / total_iters
    return average_loss


def train_nodes(model, graph, optimizer, with_val=False):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    outputs = model(graph)
    total_nodes = graph.train_mask.sum().item()
    loss = criterion(outputs[graph.train_mask], graph.y[graph.train_mask]) * total_nodes

    if with_val:
        val_nodes = graph.val_mask.sum().item()
        total_nodes += val_nodes
        loss += criterion(outputs[graph.val_mask], graph.y[graph.val_mask]) * val_nodes

    loss = loss / total_nodes

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train_nodes_PPI(model, graphs, optimizer, val_graphs=None):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    loss = 0
    total_nodes = 0
    num_classes = graphs[0].y.size()[1]
    for g in graphs:
        outputs = model(g)
        for c in range(num_classes):
            loss += criterion(outputs[:, [2 * c, 2 * c + 1]], g.y[:, c].long()) * g.num_nodes
        total_nodes += g.num_nodes

    if val_graphs is not None:
        for g in val_graphs:
            outputs = model(g)
            for c in range(num_classes):
                loss += criterion(outputs[:, [2 * c, 2 * c + 1]], g.y[:, c].long()) * g.num_nodes
            total_nodes += g.num_nodes

    loss = loss / total_nodes

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def separate_data(dataset: InMemoryDataset, seed, fold_idx):
    assert 0 <= fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    idx_list = []
    for idx in skf.split(np.zeros(len(dataset)), np.zeros(len(dataset))):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return dataset[list(train_idx)], dataset[list(test_idx)]
