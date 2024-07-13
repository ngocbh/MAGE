import torch

from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from shap.maskers import Masker

from mage.utils.validation import check_mask_input
from mage.utils.gnn_helpers import to_networkx


"""
Graph building/Perturbation
"""


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through masking the unselected nodes with zero features"""
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through splitting the selected nodes from the original graph"""
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


class PyGDataMasker(Masker):
    def __init__(self, method: str = 'split'):
        self.method = method
        self.masking_fn = build_methods[method]
        super(PyGDataMasker, self).__init__()

    def __call__(self, mask, input):
        mask = check_mask_input(mask, size=input.num_nodes)
        mask = torch.from_numpy(mask).float().to(input.x.device)
        masked_x, masked_edge_index = self.masking_fn(input.x, input.edge_index, mask)
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)
        return masked_data

    def build_rag(self, input, num_segments=-1):
        num_segments = num_segments if num_segments != -1 else input.num_nodes
        graph = to_networkx(input, to_undirected=True)
        return graph
