import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import pickle
from datetime import datetime
import os.path as osp
import os
import argparse


def generate_csbm(n, p, mu1, mu2, d, h):
    # avg. number of neignbors with same/different class
    c_in = d * h
    c_out = d - c_in

    # label, half +1, half -1, will be processed to {1, 0} later
    y = np.ones(n)
    y[(n // 2):] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] * y[j] > 0:
                Flip = np.random.binomial(1, c_in / n)
            else:
                Flip = np.random.binomial(1, c_out / n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.ones([1, p]) * (1 / np.sqrt(p))

    for i in range(n // 2):  # where y == 1
        Z = np.random.normal(0, 1, [1, p])
        x[i] = mu1 * u + Z / np.sqrt(p)

    for i in range(n // 2, n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = mu2 * u + Z / np.sqrt(p)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))
    # order edge list and remove duplicates if any.
    data.coalesce()

    return data


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    if file name not specified, use time stamp.
    '''
    now = datetime.now()
    surfix = now.strftime('%b_%d_%Y-%H:%M')
    if file_name is None:
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        tmp_data_name = file_name
    p2cSBM_data = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2cSBM_data, 'bw') as f:
        pickle.dump(data, f)
    return p2cSBM_data


class CSBM(InMemoryDataset):

    def __init__(self, root, num_nodes, dim_feat, mu1, mu2, d, h,
                 transform=None, pre_transform=None):

        name = f'csbm_n_{num_nodes}_p_{dim_feat}_mu_{mu1}_{mu2}_d_{d}_h_{h}.pkl'
        self.name = name
        root = osp.join(root, 'csbm', self.name[:-4])

        self.parameters = (num_nodes, dim_feat, mu1, mu2, d, h)

        super(CSBM, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.raw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                tmp_data = generate_csbm(*self.parameters)

                _ = save_data_to_pickle(tmp_data,
                                        p2root=self.raw_dir,
                                        file_name=self.name)
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
