import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph


"""
Graph building/Perturbation
`graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
"""


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through masking the unselected nodes with zero features"""
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through spliting the selected nodes from the original graph"""
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index


def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through removing the unselected nodes from the original graph"""
    ret_X = X[node_mask == 1]
    ret_edge_index, _ = subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
    return ret_X, ret_edge_index


build_methods = {
    "zero_filling": graph_build_zero_filling,
    "split": graph_build_split,
    "remove": graph_build_remove,
}


class MaskedDataset(Dataset):
    def __init__(self, data, masks, subgraph_building_method):
        super().__init__()

        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y

        if not torch.is_tensor(masks):
            masks = torch.tensor(masks)

        self.masks = masks.type(torch.float32).to(self.device)
        self.subgraph_building_func = build_methods[subgraph_building_method]

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(
            self.x, self.edge_index, self.masks[idx]
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)

        return masked_data


def get_reward_func_for_gnn_nc(model, node_idx, target_class, payoff_type="prob"):
    """
        get prediction function for GNN model with graph classification task
    """
    def char_func(data):
        with torch.no_grad():
            if len(data.x) == 0:
                return torch.tensor(0)
            logits = model(data=data)
            probs = F.softmax(logits, dim=-1)
            batch_size = data.batch.max() + 1

            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            logits = logits.reshape(batch_size, -1, probs.shape[-1])

            if "logit" in payoff_type:
                payoff = logits[:, node_idx, target_class]
            elif "log_prob" in payoff_type:
                payoff = logits.log_softmax(dim=-1)[:, node_idx, target_class]
            elif "prob" in payoff_type:
                payoff = logits.softmax(dim=-1)[:, node_idx, target_class]
            else:
                raise ValueError(f"unknown payoff type: {payoff_type}")
        return payoff

    return char_func


def get_reward_func_for_gnn_gc(model, target_class, payoff_type="prob"):
    """
        get prediction function for GNN model with graph classification task
    """
    def char_func(data):
        with torch.no_grad():
            if len(data.x) == 0:
                return torch.tensor(0)
            logits = model(data=data)
            if "logit" in payoff_type:
                payoff = logits[:, target_class]
            elif "log_prob" in payoff_type:
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            elif "prob" in payoff_type:
                payoff = logits.softmax(dim=-1)[:, target_class]
            else:
                raise ValueError(f"unknown payoff type: {payoff_type}")
        return payoff

    return char_func


def get_reward_func_for_point_cloud(model, target_class, payoff_type="prob"):
    """
        get prediction function for GNN model with graph classification task
    """
    def char_func(data):
        with torch.no_grad():
            logits = model(data=data)
            if "logit" in payoff_type:
                payoff = logits[:, target_class]
            elif "log_prob" in payoff_type:
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            elif "prob" in payoff_type:
                payoff = logits.softmax(dim=-1)[:, target_class]
            else:
                raise ValueError(f"unknown payoff type: {payoff_type}")
        return payoff

    return char_func


def get_reward_func_for_image_classification(model, target_class, payoff_type="prob"):
    def char_func(data):
        with torch.no_grad():
            logits = model(data)
            if "logit" in payoff_type:
                payoff = logits[:, target_class]
            elif "log_prob" in payoff_type:
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            elif "prob" in payoff_type:
                payoff = logits.softmax(dim=-1)[:, target_class]
            else:
                raise ValueError(f"unknown payoff type: {payoff_type}")
        return payoff

    return char_func


def normalize_reward_func(fn, norm):
    def char_func(data):
        return fn(data) - norm

    return char_func


def to_networkx(
    data,
    node_index=None,
    node_attrs=None,
    edge_attrs=None,
    to_undirected=False,
    remove_self_loops=False,
):
    r"""
    Extend the PyG to_networkx with extra node_index argument, so subgraphs can be plotted with correct ids

    Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)


        node_index (iterable): Pass in it when there are some nodes missing.
                 max(node_index) == max(data.edge_index)
                 len(node_index) == data.num_nodes
    """
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    if node_index is not None:
        """
        There are some nodes missing. The max(data.edge_index) > data.x.shape[0]
        """
        G.add_nodes_from(node_index)
    else:
        G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def sparsity(coalition: list, data: Data, subgraph_building_method='zero_filling'):
    if subgraph_building_method in ['zero_filling', 'remove']:
        return 1.0 - len(coalition) / data.num_nodes
    elif subgraph_building_method == 'split':
        row, col = data.edge_index
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[coalition] = 1.0
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        return 1.0 - (edge_mask.sum() / edge_mask.shape[0]).item()


def _dfs(u, adj_lst, visited, subset, temp_lst):
    visited[u] = True
    temp_lst.append(u)

    for v in adj_lst[u].keys():
        if not visited[v]:
            temp_lst = _dfs(v, adj_lst, visited, subset, temp_lst)
    return temp_lst


def subgraph_connected_components(adj_lst, subset):
    """
        Given a graph represented by a adjacency list and a subset of vertices,
        return connected components of the subgraph induced by the subset.
    """
    num_nodes = len(adj_lst)
    visited = [True] * num_nodes
    for u in subset:
        visited[u] = False
    components = []
    for u in subset:
        if not visited[u]:
            cpn = []
            cpn = _dfs(u, adj_lst, visited, subset, cpn)
            components.append(cpn)

    return components
        

def get_neighbors(G: nx.Graph, subset):
    neighbors = set()
    for node in subset:
        neighbors.update(G.neighbors(node))
    neighbors = neighbors - set(subset)
    return neighbors
