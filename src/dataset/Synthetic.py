import os
import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch_sparse
import torch
from torch_geometric.data import Data, InMemoryDataset


class SynCora(InMemoryDataset):

    def __init__(self, root, homophily, seed=1, transform=None, pre_transform=None):
        self.root = str(os.path.join(root, 'syn-cora'))

        self.name = str(f'h{homophily:.2f}-r{seed}.npz')
        self.name2 = str(f'h{homophily:.2f}-r{seed}.pt')

        print(self.name2)
        print(self.root)

        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        print(self.root)

        super(SynCora, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name2, ]

    def process(self):
        adj, features, labels = load_npz(file_name=os.path.join(self.root, self.name))
        x = torch.FloatTensor(features.toarray())
        edge_index = torch_sparse.from_scipy(adj)[0]
        y = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list = [data, ]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SynProducts(InMemoryDataset):

    def __init__(self, root, homophily, seed=1, transform=None, pre_transform=None):
        self.root = str(os.path.join(root, 'syn-products'))

        self.name = str(f'h{homophily:.2f}-r{seed}.npz')
        self.name2 = str(f'h{homophily:.2f}-r{seed}.pt')

        print(self.name2)
        print(self.root)

        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        print(self.root)

        super(SynProducts, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name2, ]

    def process(self):
        adj, features, labels = load_npz(file_name=os.path.join(self.root, self.name))
        x = torch.FloatTensor(features.toarray())
        edge_index = torch_sparse.from_scipy(adj)[0]
        y = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list = [data, ]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        print(list(loader.keys()))
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels


class SynCora_Masked(InMemoryDataset):

    def __init__(self, root, homophily, seed=1, transform=None, pre_transform=None):
        self.root = str(os.path.join(root, 'syn-cora'))

        self.name = str(f'h{homophily:.2f}-r{seed}.npz')
        self.name2 = str(f'h{homophily:.2f}-r{seed}.pt')

        print(f"Loading {self.name2} from {self.root}")

        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        super(SynCora_Masked, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        features = self.data.x
        labels = self.data.y
        base = 1.001 ** torch.arange(features.shape[1])
        expsum = features @ base
        idxs = torch.argsort(expsum)
        labels_sorted = labels[idxs]

        train_mask_sorted = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask_sorted = torch.ones(features.shape[0], dtype=torch.bool)

        # for each class, use 25% as the training set, and 75% as the testing

        for label in range(5):
            label_subset = torch.where(labels_sorted == label)[0]
            ii = list(idx.item() for i, idx in enumerate(label_subset) if i % 4 == 0)
            train_mask_sorted[ii] = True
            test_mask_sorted[ii] = False

        # convert mask back to original order
        self.data.train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        self.data.test_mask = torch.zeros(features.shape[0], dtype=torch.bool)

        self.data.train_mask[idxs[train_mask_sorted]] = True
        self.data.test_mask[idxs[test_mask_sorted]] = True

    @property
    def processed_file_names(self):
        return [self.name2, ]

    def process(self):
        adj, features, labels = load_npz(file_name=os.path.join(self.root, self.name))
        x = torch.FloatTensor(features.toarray())
        edge_index = torch_sparse.from_scipy(adj)[0]
        y = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list = [data, ]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SynProducts_Masked(InMemoryDataset):

    def __init__(self, root, homophily, seed=1, transform=None, pre_transform=None):
        self.root = str(os.path.join(root, 'syn-products'))

        self.name = str(f'h{homophily:.2f}-r{seed}.npz')
        self.name2 = str(f'h{homophily:.2f}-r{seed}.pt')

        print(self.name2)
        print(self.root)

        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        print(self.root)

        super(SynProducts_Masked, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        features = self.data.x
        labels = self.data.y
        base = 1.001 ** torch.arange(features.shape[1])
        expsum = features @ base
        idxs = torch.argsort(expsum)
        labels_sorted = labels[idxs]

        train_mask_sorted = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask_sorted = torch.ones(features.shape[0], dtype=torch.bool)

        for label in range(10):
            label_subset = torch.where(labels_sorted == label)[0]
            ii = list(idx.item() for i, idx in enumerate(label_subset) if i % 4 == 0)
            train_mask_sorted[ii] = True
            test_mask_sorted[ii] = False

        # convert mask back to original order
        self.data.train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        self.data.test_mask = torch.zeros(features.shape[0], dtype=torch.bool)

        self.data.train_mask[idxs[train_mask_sorted]] = True
        self.data.test_mask[idxs[test_mask_sorted]] = True

    @property
    def processed_file_names(self):
        return [self.name2, ]

    def process(self):
        adj, features, labels = load_npz(file_name=os.path.join(self.root, self.name))
        x = torch.FloatTensor(features.toarray())
        edge_index = torch_sparse.from_scipy(adj)[0]
        y = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list = [data, ]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        print(list(loader.keys()))
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels
