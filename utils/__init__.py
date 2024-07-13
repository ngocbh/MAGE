import json
import logging
import pytz
import os
import torch
import random
import numpy as np
import scipy.sparse as sp
import torch_geometric as tg
from abc import ABC
from datetime import datetime
from typing import List, Union
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from collections import defaultdict
import torch.nn as nn


def padding_graphs(adj, max_num_nodes):
    num_nodes = adj.shape[0]
    adj_padded = np.eye((max_num_nodes))
    adj_padded[:num_nodes, :num_nodes] = adj.cpu()
    return torch.tensor(adj_padded, dtype=torch.long)


def padding_features(features, max_num_nodes):
    feat_dim = features.shape[1]
    num_nodes = features.shape[0]
    features_padded = np.zeros((max_num_nodes, feat_dim))
    features_padded[:num_nodes] = features.cpu()
    return torch.tensor(features_padded, dtype=torch.float)


def padded_datalist(data_list, adj_list, max_num_nodes):
    for i, data in enumerate(data_list):
        data.adj_padded = padding_graphs(adj_list[i], max_num_nodes)
        data.x_padded = padding_features(data.x, max_num_nodes)
    return data_list


def from_adj_to_edge_index(adj):
    A = sp.csr_matrix(adj)
    edges, edge_weight = tg.utils.from_scipy_sparse_matrix(A)
    return edges, edge_weight


def from_edge_index_to_adj(edge_index, edge_weight, max_n):
    adj = tg.utils.to_scipy_sparse_matrix(edge_index, edge_attr=edge_weight).toarray()
    assert len(adj) <= max_n, "The adjacency matrix contains more nodes than the graph!"
    if len(adj) < max_n:
        adj = np.pad(adj, (0, max_n - len(adj)), mode="constant")
    return torch.FloatTensor(adj)


def from_edge_index_to_sparse_adj(edge_index, edge_weight, max_n):
    adj = sp.coo_matrix(
        (edge_weight, (edge_index[0, :], edge_index[1, :])),
        shape=(max_n, max_n),
        dtype=np.float32,
    )
    return adj


def from_adj_to_edge_index_torch(adj):
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    # if adj.requires_grad:
    # edge_index.requires_grad = True
    # edge_attr.requires_grad = True
    return edge_index, edge_attr


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()


def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO, mode='w'):
    check_dir(log_path)

    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(os.path.join(log_path, log_file), mode=mode)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def fix_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def perturb_input(data, hard_edge_mask, subset):
    """add 2 additional empty node into the motif graph"""
    num_add_node = 2
    num_perturb_graph = 10
    subgraph_x = data.x[subset]
    subgraph_edge_index = data.edge_index[:, hard_edge_mask]
    row, col = data.edge_index

    mappings = row.new_full((data.num_nodes,), -1)
    mappings[subset] = torch.arange(subset.size(0), device=row.device)
    subgraph_edge_index = mappings[subgraph_edge_index]

    subgraph_y = data.y[subset]

    num_node_subgraph = subgraph_x.shape[0]

    # add two nodes to the subgraph, the node features are all 0.1
    subgraph_x = torch.cat([subgraph_x,
                            torch.ones(2, subgraph_x.shape[1]).to(subgraph_x.device)],
                           dim=0)
    subgraph_y = torch.cat([subgraph_y,
                            torch.zeros(num_add_node).type(torch.long).to(subgraph_y.device)], dim=0)

    perturb_input_list = []
    for _ in range(num_perturb_graph):
        to_node = torch.randint(0, num_node_subgraph, (num_add_node,))
        frm_node = torch.arange(num_node_subgraph, num_node_subgraph + num_add_node, 1)
        add_edges = torch.cat([torch.stack([to_node, frm_node], dim=0),
                               torch.stack([frm_node, to_node], dim=0),
                               torch.stack([frm_node, frm_node], dim=0)], dim=1)
        perturb_subgraph_edge_index = torch.cat([subgraph_edge_index,
                                                 add_edges.to(subgraph_edge_index.device)], dim=1)
        perturb_input_list.append(Data(x=subgraph_x, edge_index=perturb_subgraph_edge_index, y=subgraph_y))

    return perturb_input_list


class Recorder(ABC):
    def __init__(self, recorder_filename):
        # init the recorder
        self.recorder_filename = recorder_filename
        if os.path.isfile(recorder_filename):
            with open(recorder_filename, 'r') as f:
                self.recorder = json.load(f)
        else:
            self.recorder = {}
            check_dir(os.path.dirname(recorder_filename))

    @classmethod
    def load_and_change_dict(cls, ori_dict, experiment_settings, experiment_data):
            key = experiment_settings[0]
            if key not in ori_dict.keys():
                ori_dict[key] = {}
            if len(experiment_settings) == 1:
                ori_dict[key] = experiment_data
            else:
                ori_dict[key] = cls.load_and_change_dict(ori_dict[key],
                                                         experiment_settings[1:],
                                                         experiment_data)
            return ori_dict

    def append(self, experiment_settings, experiment_data):
        ex_dict = self.recorder

        self.recorder = self.load_and_change_dict(ori_dict=ex_dict,
                                                  experiment_settings=experiment_settings,
                                                  experiment_data=experiment_data)

    def save(self):
        with open(self.recorder_filename, 'w') as f:
            json.dump(self.recorder, f, indent=2)




def control_sparsity(mask: torch.Tensor, sparsity: float=None):
    r"""
    Transform the mask where top 1 - sparsity values are set to inf.
    Args:
        mask (torch.Tensor): Mask that need to transform.
        sparsity (float): Sparsity we need to control i.e. 0.7, 0.5 (Default: :obj:`None`).
    :rtype: torch.Tensor
    """
    if sparsity is None:
        sparsity = 0.7

    # Not apply here, Please refer to specific explainers in other directories
    #
    # if data_args.model_level == 'node':
    #     assert self.hard_edge_mask is not None
    #     mask_indices = torch.where(self.hard_edge_mask)[0]
    #     sub_mask = mask[self.hard_edge_mask]
    #     mask_len = sub_mask.shape[0]
    #     _, sub_indices = torch.sort(sub_mask, descending=True)
    #     split_point = int((1 - sparsity) * mask_len)
    #     important_sub_indices = sub_indices[: split_point]
    #     important_indices = mask_indices[important_sub_indices]
    #     unimportant_sub_indices = sub_indices[split_point:]
    #     unimportant_indices = mask_indices[unimportant_sub_indices]
    #     trans_mask = mask.clone()
    #     trans_mask[:] = - float('inf')
    #     trans_mask[important_indices] = float('inf')
    # else:
    _, indices = torch.sort(mask, descending=True)
    mask_len = mask.shape[0]
    split_point = int((1 - sparsity) * mask_len)
    important_indices = indices[: split_point]
    unimportant_indices = indices[split_point:]
    trans_mask = mask.clone()
    trans_mask[important_indices] = float('inf')
    trans_mask[unimportant_indices] = - float('inf')

    return trans_mask


class XCollector:
    r"""
    XCollector is a data collector which takes processed related prediction probabilities to calculate Fidelity+
    and Fidelity-.

    Args:
        sparsity (float): The Sparsity is use to transform the soft mask to a hard one.

    .. note::
        For more examples, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    """

    def __init__(self):
        self.__related_preds = defaultdict(list)

    def new(self):
        r"""
        Clear class members.
        """
        self.__related_preds = defaultdict(list), []

    def collect_data(self, related_preds: dict):
        for key, value in related_preds.items():
            self.__related_preds[key].append(value)

    def get_average(self, metric):
        if metric not in self.__related_preds.keys():
            raise ValueError(f"no metric: {metric}")

        score = np.array(self.__related_preds[metric])
        return score.mean().item(), score.std().item()

    def get_summarized_results(self):
        ret = {}
        for metric in self.__related_preds.keys():
            ret[metric] = self.get_average(metric)
        return ret


class ExplanationProcessor(nn.Module):
    r"""
    Explanation Processor is edge mask explanation processor which can handle sparsity control and use
    data collector automatically.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        device (torch.device): Specify running device: CPU or CUDA.

    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.edge_mask = None
        self.model = model
        self.device = device
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = False

    def eval_related_pred(self, x: torch.Tensor, edge_index: torch.Tensor, masks: List[torch.Tensor], **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx # graph level: 0, node level: node_idx

        related_preds = []

        for label, mask in enumerate(masks):
            # origin pred
            for edge_mask in self.edge_mask:
                edge_mask.data = float('inf') * torch.ones(mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            for edge_mask in self.edge_mask:
                edge_mask.data = mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            for edge_mask in self.edge_mask:
                edge_mask.data = - mask
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            for edge_mask in self.edge_mask:
                edge_mask.data = - float('inf') * torch.ones(mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # Store related predictions for further evaluation.
            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            related_preds[label] = {key: pred.softmax(0)[label].item()
                                    for key, pred in related_preds[label].items()}

        return related_preds

    def forward(self, data: Data, masks: List[torch.Tensor], x_collector: XCollector, **kwargs):
        r"""
        Please refer to the main function in `metric.py`.
        """

        data.to(self.device)
        node_idx = kwargs.get('node_idx')
        y_idx = 0 if node_idx is None else node_idx

        assert not torch.isnan(data.y[y_idx].squeeze())

        self.num_edges = data.edge_index.shape[1]
        self.num_nodes = data.x.shape[0]

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(data.x, data.edge_index, masks, **kwargs)

        x_collector.collect_data(masks,
                                 related_preds,
                                 data.y[y_idx].squeeze().long().item())
