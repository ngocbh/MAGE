# https://github.com/divelab/DIG/blob/dig/dig/xgraph/dataset/syn_dataset.py

import numpy as np
import torch
import networkx as nx
import pickle
import numpy as np
import os
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx
from dataset.syn_utils.gengraph import *
from torch_geometric.utils import dense_to_sparse
from utils import padded_datalist, from_edge_index_to_adj


class BaHouseGrid(InMemoryDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.
    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = "https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}"
    # Format: name: [display_name, url_name, filename]
    names = {
        "ba_house_grid": ["BA_House_Grid", "BA_House_Grid.pkl", "BA_House_Grid"],
        "ba_house_and_grid": ["BA_House_And_Grid", "BA_House_And_Grid.pkl", "BA_House_And_Grid"],
        "ba_house_or_grid": ["BA_House_Or_Grid", "BA_House_Or_Grid.pkl", "BA_House_Or_Grid"],
    }

    def __init__(
        self, root, name, transform=None, pre_transform=None,
        num_graphs=10000, num_shapes=1, width_basis=20, nnf=1,
        seed=2, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    ):
        self.name = name.lower()
        self.num_graphs = num_graphs
        self.num_shapes = num_shapes
        self.width_basis = width_basis
        self.seed = seed
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = train_ratio
        self.nnf = nnf # num node features
        super(BaHouseGrid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return f"{self.names[self.name][2]}.pkl"

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        if self.name.lower() == "BA_2Motifs".lower():
            url = self.url.format(self.names[self.name][1])
            path = download_url(url, self.raw_dir)

    def process(self):
        """Generate synthetic graohs and convert them into Pytorch geometric Data object.

        Returns:
            Data: converted synthetic Pytorch geometric Data object
        """
        if self.name.lower() == "BA_House_Grid".lower():
            # Binary graph classification task
            motifs = ["_house", "_grid"]
            labels = [0, 1]
            probs = [0.5, 0.5]

            data_list = []
            adj_list = []
            max_num_nodes = 0
            for graph_idx in range(self.num_graphs):
                idx = np.random.choice(list(range(len(motifs))), p=probs)
                name = motifs[idx]
                generate_function = "gen_ba" + name
                # print(idx, name, generate_function)
                # nb_shapes = np.random.randint(1, self.num_shapes)
                G, _, _ = eval(generate_function)(
                    nb_shapes=1,
                    width_basis=self.width_basis,
                    m=1,
                    feature_generator=featgen.ConstFeatureGen(
                        np.ones(self.nnf, dtype=float)
                    ), 
                    is_weighted=True,
                )
                data = self.from_G_to_data(G, graph_idx, labels[idx], name)
                max_num_nodes = max(max_num_nodes, data.num_nodes)
                adj = from_edge_index_to_adj(
                    data.edge_index, data.edge_attr, data.num_nodes
                )
                adj_list.append(adj)
                data_list.append(data)
            data_list = padded_datalist(data_list, adj_list, max_num_nodes)

        elif self.name.lower() in ["ba_house_and_grid", "ba_house_or_grid"]:
            # Binary graph classification task
            motifs = ["", "_house", "_grid", "_house_grid"]
            if "and" in self.name.lower():
                labels = [0, 0, 0, 1]
                probs = [0.5/3, 0.5/3, 0.5/3, 0.5]
            elif "or" in self.name.lower():
                labels = [0, 1, 1, 1]
                probs = [0.5, 0.5/3, 0.5/3, 0.5/3]

            data_list = []
            adj_list = []
            max_num_nodes = 0
            for graph_idx in range(self.num_graphs):
                idx = np.random.choice(list(range(len(motifs))), p=probs)
                name = motifs[idx]
                generate_function = "gen_ba" + name
                # print(idx, name, generate_function)
                # nb_shapes = np.random.randint(1, self.num_shapes)
                m = np.random.randint(1, 3)
                G, _, _ = eval(generate_function)(
                    nb_shapes=1,
                    width_basis=self.width_basis,
                    m=m,
                    feature_generator=featgen.ConstFeatureGen(
                        np.ones(self.nnf, dtype=float)
                    ), 
                    is_weighted=True,
                )
                data = self.from_G_to_data(G, graph_idx, labels[idx], name)
                max_num_nodes = max(max_num_nodes, data.num_nodes)
                adj = from_edge_index_to_adj(
                    data.edge_index, data.edge_attr, data.num_nodes
                )
                adj_list.append(adj)
                data_list.append(data)
            data_list = padded_datalist(data_list, adj_list, max_num_nodes)
            
        else:
            generate_function = "gen_" + self.name

            G, labels, name = eval(generate_function)(
                nb_shapes=self.num_shapes,
                width_basis=self.width_basis,
                feature_generator=featgen.ConstFeatureGen(
                    np.ones(self.nnf, dtype=float)
                ),
            )

            data = from_networkx(G.to_undirected(), all)
            data.adj = torch.LongTensor(nx.to_numpy_matrix(G))
            data.num_classes = len(np.unique(labels))
            data.y = torch.LongTensor(labels)
            data.x = data.x.float()
            data.edge_attr = torch.ones(data.edge_index.size(1))
            n = data.num_nodes
            data.train_mask, data.val_mask, data.test_mask = (
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
            )
            train_ids, test_ids = train_test_split(
                range(n),
                test_size=self.test_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            train_ids, val_ids = train_test_split(
                train_ids,
                test_size=self.val_ratio,
                random_state=self.seed,
                shuffle=True,
            )

            data.train_mask[train_ids] = 1
            data.val_mask[val_ids] = 1
            data.test_mask[test_ids] = 1

            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list = [data]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def read_syn_data(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

        x = torch.from_numpy(features).float()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        y = torch.from_numpy(np.where(y)[1])
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        data.edge_label_matrix = torch.from_numpy(edge_label_matrix)
        return data

    def gen_motif_edge_mask(self, data, node_idx=0, num_hops=3):
        if self.name in ['ba_house_grid', 'ba_house_and_grid', 'ba_house_or_grid']:
            d_em, d_ei = data.edge_mask.shape[0], data.edge_index.shape[-1]
            if d_em != d_ei:
                return torch.nn.functional.pad(data.edge_mask, (0, d_ei - d_em), 'constant', 0.)
            else:
                return data.edge_mask

    def from_G_to_data(self, G, graph_idx, label, name='_house'):
        # attr_list = [str(attr) for attr in list(nx.get_edge_attributes(G, 'weight').values())]
        attr_list = nx.get_edge_attributes(G, 'weight').values()
        data = from_networkx(G, group_edge_attrs=all)
        data.x = data.feat.float()
        # adj = torch.LongTensor(nx.to_numpy_matrix(G))
        data.y = torch.tensor(label).float().reshape(-1, 1)
        data.edge_mask = torch.squeeze(data.edge_attr)
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        data.idx = graph_idx
        return data

    def __repr__(self):
        return "{}({})".format(self.names[self.name][0], len(self))
