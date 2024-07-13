# From Discovering Invariant Rationales for Graph Neural Networks

import os.path as osp
import pickle as pkl

import yaml
import torch
import torch.nn.functional as F
import random
import numpy as np
import torch_geometric
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

try:
    from .spmotif_utils import gen_dataset
except ImportError:
    from spmotif_utils import gen_dataset


class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, name, b=0.7, transform=None, pre_transform=None, pre_filter=None):

        self.b = b
        self.name = name
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ['train.pkl', 'val.pkl', 'test.pkl']

    @property
    def processed_file_names(self):
        return ['SPMotif.pt']

    def download(self):
        print('[INFO] Generating SPMotif dataset...')
        gen_dataset(self.b, Path(self.raw_dir))

    def process(self):
        data_list = []

        for mode in self.splits:
            idx = self.raw_file_names.index('{}.pkl'.format(mode))
            edge_index_list, label_list, ground_truth_list, role_id_list, pos = pkl.load(open(osp.join(self.raw_dir, self.raw_file_names[idx]), 'rb'))
            for idx, (edge_index, y, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
                edge_index = torch.from_numpy(edge_index).long()
                node_idx = torch.unique(edge_index)
                assert node_idx.max() == node_idx.size(0) - 1
                # x = torch.zeros(node_idx.size(0), 4)
                # index = [i for i in range(node_idx.size(0))]
                # x[index, z] = 1
                x = torch.rand((node_idx.size(0), 4))
                edge_attr = torch.ones(edge_index.size(1), 1)
                y = torch.tensor(y, dtype=torch.long).reshape(-1)

                node_label = torch.tensor(z, dtype=torch.float)
                node_label[node_label != 0] = 1
                edge_label = torch.tensor(ground_truth, dtype=torch.float)

                assert edge_label.shape[-1] == edge_index.shape[-1]

                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def gen_motif_edge_mask(self, data, node_idx=0, num_hops=3):
        d_em, d_ei = data.edge_label.shape[0], data.edge_index.shape[-1]
        if d_em != d_ei:
            return torch.nn.functional.pad(data.edge_label, (0, d_ei - d_em), 'constant', 0.)
        else:
            return data.edge_label
