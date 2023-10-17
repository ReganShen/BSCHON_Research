import numpy as np
import scipy.sparse as sp
import torch
import json
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def read_json_file():
    with open('parameters.json', 'r') as file:
        data = json.load(file)
    return data


def load_data_Citeseer(path="./data/CiteSeer", dataset="CiteSeer"):
    """Load citation network dataset (CiteSeer for now)"""
    print('Loading {} dataset...'.format(dataset))


    dataset = Planetoid(root=path, name=dataset)
    data = dataset[0]

    # Convert edge_index to a scipy sparse matrix
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]),
                         (data.edge_index[0].numpy(), data.edge_index[1].numpy())))

    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = sp.csr_matrix(data.x.numpy())
    labels = encode_onehot(data.y.numpy())

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # Note: Here we're making a naive split based on node indices. This might not be ideal for all tasks.



    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    largeTrain = int(0.8 * (len(adj)))
    actualTrain = int(0.8 * largeTrain)
    idx_train = range(actualTrain)
    idx_val = range(actualTrain, largeTrain)
    idx_test = range(largeTrain, (len(adj) - 1))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_Pubmed(path="./data/Pubmed", dataset="Pubmed"):
    """Load citation network dataset (Pubmed for now)"""
    print('Loading {} dataset...'.format(dataset))

    dataset = Planetoid(root=path, name=dataset)
    data = dataset[0]

    # Convert edge_index to a scipy sparse matrix
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]),
                         (data.edge_index[0].numpy(), data.edge_index[1].numpy())))

    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = sp.csr_matrix(data.x.numpy())
    labels = encode_onehot(data.y.numpy())

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # Note: Here we're making a naive split based on node indices. This might not be ideal for all tasks.


    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])


    largeTrain = int(0.8 * (len(adj)))
    actualTrain = int(0.8 * largeTrain)
    idx_train = range(actualTrain)
    idx_val = range(actualTrain, largeTrain)
    idx_test = range(largeTrain, (len(adj) - 1))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test





def load_mutag_data(path="/tmp/MUTAG", dataset="MUTAG"):
    """Load the MUTAG dataset."""
    print('Loading {} dataset...'.format(dataset))

    dataset = TUDataset(root=path, name=dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Use first graph as an example; in real use cases, you would process each graph.
    data = dataset[0]

    # Convert edge_index to a scipy sparse matrix
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]),
                         (data.edge_index[0].numpy(), data.edge_index[1].numpy())))

    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = sp.csr_matrix(data.x.numpy())
    labels = encode_onehot(data.y.numpy())

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # Sample smaller splits, you may need to adjust this depending on MUTAG's size:
    num_nodes = adj.shape[0]
    idx_train = range(int(0.6 * num_nodes))
    idx_val = range(int(0.6 * num_nodes), int(0.8 * num_nodes))
    idx_test = range(int(0.8 * num_nodes), num_nodes)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test