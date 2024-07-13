import networkx as nx
import numpy as np

from shap.maskers import Masker
from shap.utils._exceptions import DimensionError

from mage.utils.validation import check_graph_input, check_mask_input


class NxMasker(Masker):
    def __init__(self, method: str = 'remove'):
        self.method = method

    def __call__(self, mask, input: nx.Graph):
        input = check_graph_input(input)
        mask = check_mask_input(mask, size=input.number_of_nodes())

        if self.method == 'remove':
            selected_nodes = np.where(mask == 1)[0]

            return input.subgraph(selected_nodes)

    def to_networkx(self, input: nx.Graph):
        return input
