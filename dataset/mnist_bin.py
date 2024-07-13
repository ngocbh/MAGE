# adapt from https://github.com/bknyaz/graph_attention_pool/blob/master/graphdata.py

import os
import numpy as np
import os.path as osp
import pickle
import zipfile
import torch
import torch.utils
import torch.utils.data

from scipy.spatial.distance import cdist
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data, download_url

from pathlib import Path
from torch_geometric.loader import DataLoader
from utils import padded_datalist, from_edge_index_to_adj


def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(-dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


class MNIST75sp_Binary(InMemoryDataset):
    url = 'https://drive.google.com/u/0/uc?id=1SF6lWf9kFIffjMgQGwJzTOICbYlkLFvh&export=download&confirm=t&uuid=459edb3c-f38e-45e3-905c-849c493aa346&at=AB6BwCDLavNQUgdmoIeX2eAB4Aln:1696868111003'
    splits = ["test", "train"]

    def __init__(
        self,
        root,
        name,
        use_mean_px=True,
        use_coord=True,
        node_gt_att_threshold=-1,
        max_num_gt_sp=15,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.node_gt_att_threshold = node_gt_att_threshold
        self.use_mean_px, self.use_coord = use_mean_px, use_coord
        self.max_num_gt_sp = max_num_gt_sp
        self.name = name.lower()
        super(MNIST75sp_Binary, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        # idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["mnist_75sp_train.pkl", "mnist_75sp_test.pkl",
                "mnist_75sp_train_superpixels.pkl",
                "mnist_75sp_test_superpixels.pkl"]

    @property
    def processed_file_names(self):
        return ["mnist75sp.pt"]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        data_list = []

        for mode in self.splits:
            data_file = 'mnist_75sp_%s.pkl' % mode
            with open(osp.join(self.raw_dir, data_file), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)

            sp_file = 'mnist_75sp_%s_superpixels.pkl' % mode
            with open(osp.join(self.raw_dir, sp_file), 'rb') as f:
                self.all_superpixels = pickle.load(f)

            self.use_mean_px = self.use_mean_px
            self.use_coord = self.use_coord
            self.n_samples = len(self.labels)
            self.img_size = 28
            self.node_gt_att_threshold = self.node_gt_att_threshold

            self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []

            for index, sample in enumerate(self.sp_data):
                mean_px, sp_coord, sp_order = sample[:3]
                superpixels = self.all_superpixels[index]
                coord = sp_coord / self.img_size
                A = compute_adjacency_matrix_images(coord)
                N_nodes = A.shape[0]

                A = torch.FloatTensor((A > 0.1) * A)
                edge_index, edge_attr = dense_to_sparse(A)

                x = None
                if self.use_mean_px:
                    x = mean_px.reshape(N_nodes, -1)
                if self.use_coord:
                    coord = coord.reshape(N_nodes, 2)
                    if self.use_mean_px:
                        x = np.concatenate((x, coord), axis=1)
                    else:
                        x = coord
                if x is None:
                    x = np.ones(N_nodes, 1)  # dummy features

                # replicate features to make it possible to test on colored images
                x = np.pad(x, ((0, 0), (2, 0)), 'edge')
                if self.node_gt_att_threshold == 0:
                    node_gt_att = (mean_px > 0).astype(np.float32)
                elif self.node_gt_att_threshold == -1:
                    # get 15 brightest superpixels as ground truth
                    num_gt_nodes = min(self.max_num_gt_sp, np.count_nonzero(mean_px))
                    node_gt_att_threshold = sorted(mean_px)[-num_gt_nodes]
                    node_gt_att = mean_px.copy()
                    node_gt_att[node_gt_att < node_gt_att_threshold] = 0
                else:
                    node_gt_att = mean_px.copy()
                    node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

                node_gt_att = torch.LongTensor(node_gt_att > 0).view(-1)
                row, col = edge_index
                # print(edge_index)
                # print(edge_index.shape)
                edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

                data_list.append(
                    Data(
                        x=torch.tensor(x),
                        y=torch.LongTensor([self.labels[index]]),
                        edge_index=edge_index,
                        edge_attr=edge_attr.reshape(-1, 1),
                        node_label=node_gt_att.float(),
                        edge_mask=edge_gt_att.float(),
                        sp_order=torch.tensor(sp_order),
                        sp_intensity=torch.tensor(mean_px),
                        sp_coord=torch.tensor(sp_coord),
                        superpixels=torch.tensor(superpixels),
                        name=f'MNISTSP-{mode}-{index}',
                        idx=index,
                        img_size=self.img_size
                    )
                )
                
                # if index == 33277:
                #     import matplotlib.pyplot as plt
                #     fig, ax = plt.subplots()
                #     img = np.zeros((self.img_size, self.img_size))
                #     print(mean_px)
                #     print(node_gt_att)
                #     for j, (sp_intens, sp_index) in enumerate(zip(node_gt_att, sp_order)):
                #         mask = (superpixels == sp_index)
                #         x = (mean_px[j] - 0.11) / 0.27 if sp_intens > 0 else 0
                #         img[mask] = x

                #     ax.imshow(img, cmap='gray')
                #     plt.axis('off')
                #     plt.savefig(f'example_{index}.png')
                #     raise ValueError
        idx = self.processed_file_names.index('mnist75sp.pt')
        torch.save(self.collate(data_list), self.processed_paths[idx])

    def gen_motif_edge_mask(self, data, node_idx=0, num_hops=3):
        d_em, d_ei = data.edge_mask.shape[0], data.edge_index.shape[-1]
        if d_em != d_ei:
            return torch.nn.functional.pad(data.edge_mask, (0, d_ei - d_em), 'constant', 0.)
        else:
            return data.edge_mask