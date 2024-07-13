"""
The shapely.py is brought from subgraphX in DIG library
https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py
"""
import copy
import torch
import numpy as np
from typing import Union
from scipy.special import comb
from itertools import combinations
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch, Dataset, DataLoader
from functools import partial
from collections import Counter
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
import math 
import networkx as nx

def value_func_decorator(value_func):
    """ input: list of the connected graph (X, A)
        return sum of the value of all connected graphs
    """
    def value_sum_func(batch, target_class):
        with torch.no_grad():
            logits, probs, _ = value_func(batch)
            score = probs[:, target_class]
        return score

    return value_sum_func


def GnnNets_GC2value_func(gnnNets, target_class):
    def value_func(batch):
        with torch.no_grad():
            prob = gnnNets(batch)
            score = torch.nn.functional.softmax(prob, dim=-1)
            score = score[:, target_class]
        return score
    return value_func

def eval_metric(original_score, gnn_score, sparsity):
    fidelity_score = (original_score - gnn_score) / original_score
    if isinstance(fidelity_score, torch.Tensor):
        fidelity_score = fidelity_score.item()
    score = fidelity_score * sparsity
    return fidelity_score, score

def GnnNets_NC2value_func(gnnNets_NC, node_idx: Union[int, torch.tensor], target_class: torch.tensor):
    def value_func(data):
        with torch.no_grad():
            prob = gnnNets_NC(data)
            s = prob.shape[-1]
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = data.batch.max() + 1
            prob = prob.reshape(batch_size, -1, s)
            score = prob[:, node_idx, target_class]
            return score
    return value_func


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    elif build_method.lower() == 'pc_baseline':
        return pc_common.graph_build_func
    else:
        raise NotImplementedError


class MarginalSubgraphDataset(Dataset):

    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


def marginal_contribution(data: Data, exclude_mask: np.array, include_mask: np.array,
                          value_func, subgraph_build_func):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """
    marginal_subgraph_dataset = MarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
    dataloader = DataLoader(marginal_subgraph_dataset, batch_size=256, shuffle=False, pin_memory=False, num_workers=0) # pin_memory: True, CUDA out of memory

    marginal_contribution_list = []

    for exclude_data, include_data in dataloader:
        exclude_values = value_func(exclude_data)
        include_values = value_func(include_data)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def graph_build_zero_filling(X, edge_index, node_mask: np.array):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: np.array):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def l_shapley(coalition: list, data: Data, local_raduis: int,
              value_func: str, subgraph_building_method='zero_filling'):
    """ shapley value where players are local neighbor nodes """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    set_exclude_masks = []
    set_include_masks = []
    nodes_around = [node for node in local_region if node not in coalition]
    num_nodes_around = len(nodes_around)

    for subset_len in range(0, num_nodes_around + 1):
        node_exclude_subsets = combinations(nodes_around, subset_len)
        for node_exclude_subset in node_exclude_subsets:
            set_exclude_mask = np.ones(num_nodes)
            set_exclude_mask[local_region] = 0.0
            if node_exclude_subset:
                set_exclude_mask[list(node_exclude_subset)] = 1.0
            set_include_mask = set_exclude_mask.copy()
            set_include_mask[coalition] = 1.0

            set_exclude_masks.append(set_exclude_mask)
            set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    num_players = len(nodes_around) + 1
    num_player_in_set = num_players - 1 + len(coalition) - (1 - exclude_mask).sum(axis=1)
    p = num_players
    S = num_player_in_set
    coeffs = torch.tensor(1.0 / comb(p, S) / (p - S + 1e-6))

    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    l_shapley_value = (marginal_contributions.squeeze().cpu() * coeffs).sum().item()
    return l_shapley_value


def mc_shapley(coalition: list, data: Data,
               value_func: str, subgraph_building_method='zero_filling',
               sample_num=1000) -> float:
    """ monte carlo sampling approximation of the shapley value """
    subset_build_func = get_graph_build_func(subgraph_building_method)

    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []

    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in node_indices if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.zeros(num_nodes)
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(data, exclude_mask, include_mask, value_func, subset_build_func)
    mc_shapley_value = marginal_contributions.mean().item()

    return mc_shapley_value


def mc_l_shapley(coalition: list, data: Data, local_raduis: int,
                 value_func: str, subgraph_building_method='zero_filling',
                 sample_num=1000) -> float:
    """ monte carlo sampling approximation of the l_shapley value """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def gnn_score(coalition: list, data: Data, value_func: str,
              subgraph_building_method='zero_filling', device='cpu') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = value_func(mask_data)
    # get the score of predicted class for graph or specific node idx
    return score.item()


def NC_mc_l_shapley(coalition: list, data: Data, local_raduis: int,
                    value_func: str, node_idx: int=-1, subgraph_building_method='zero_filling', sample_num=1000) -> float:
    """ monte carlo approximation of l_shapley where the target node is kept in both subgraph """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        if node_idx != -1:
            set_exclude_mask[node_idx] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0  # include the node_idx

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


class MCTSNode():

    def __init__(self, coalition: list, data: Data,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n): 
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


class MCTS():
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, n_rollout: int,
                 min_atoms: int, c_puct: float, expand_atoms: int, score_func, high2low):
        """ graph is a networkX graph """
        self.X = X
        self.edge_index = edge_index
        self.data = Data(x=self.X, edge_index=self.edge_index)
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0])
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)
        ### Modified
        self.root = self.MCTSNodeClass([])
        self.root_coalition = [i for i in range(self.num_nodes)]
        ###
        self.state_map = {str(self.root.coalition): self.root}

    def mcts_rollout(self, tree_node):
        unvisited_graph_coalition = [i for i in range(self.num_nodes) if i not in tree_node.coalition]
        if len(tree_node.coalition) >= self.min_atoms or len(tree_node.coalition)/self.num_nodes >= 0.8:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            if len(tree_node.coalition) == 0:
                node_degree_list = list(self.graph.subgraph(unvisited_graph_coalition).degree)
                node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
                all_nodes = [x[0] for x in node_degree_list]

                if len(all_nodes) < self.expand_atoms:
                    expand_nodes = all_nodes
                else:
                    expand_nodes = all_nodes[:self.expand_atoms]
            else:
                expand_nodes = []
                for n in tree_node.coalition:
                    nbrs = self.graph.adj[n]
                    for nbr, _ in nbrs.items():
                        if nbr in unvisited_graph_coalition:
                            expand_nodes.append(nbr)

            for each_node in expand_nodes:
                subgraph_coalition = [node for node in range(self.num_nodes) if node in tree_node.coalition or node == each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                
                assert len(subgraphs) == 1
                main_sub = subgraphs[0]
                for sub in subgraphs:
                    if sub.number_of_nodes() > main_sub.number_of_nodes():
                        main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Counter(new_node.coalition) == Counter(tree_node.coalition):
                    Find_same_child = True

                if Find_same_child == False:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        if len(tree_node.children) == 0:
            return tree_node.P
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"[Initialization] Rollout {rollout_idx}, {len(self.state_map)} states that have been explored.")

        explanations = []
        for _, node in self.state_map.items():
            node.coalition = [k for k in self.root_coalition if k in node.coalition]
            explanations.append(node)
            pass
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


def reward_func(reward_args, value_func, node_idx=-1, subgraph_building_method=None):
    if reward_args.reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_args.reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=reward_args.sample_num)

    elif reward_args.reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_args.reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=reward_args.sample_num)
    elif reward_args.reward_method.lower() == 'nc_mc_l_shapley':
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=reward_args.sample_num)
    else:
        raise NotImplementedError

class exploration_MCTSNode():

    def __init__(self, coalition: list, data: Data, candidates: [list], 
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.candidates = candidates
        self.coalition = coalition  # permutation of any candidates
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n): 
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


class exploration_MCTS():
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, candidates, 
                 n_rollout: int, explanation_size: int, c_puct: float, score_func):
        """ graph is a networkX graph """
        self.X = X
        self.candidates = candidates
        self.edge_index = edge_index
        self.data = Data(x=self.X, edge_index=self.edge_index)
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0])
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.num_candidates = len(candidates)
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.explanation_size = explanation_size
        self.c_puct = c_puct

        self.MCTSNodeClass = partial(exploration_MCTSNode, data=self.data, candidates=self.candidates,
                                     ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass([])
        self.state_map = {str(self.root.coalition): self.root}

    def mcts_rollout(self, tree_node):
        unvisited_candidates = [i for i in range(self.num_candidates) if i not in tree_node.coalition]
        current_explanation = []
        for substructure in tree_node.coalition:
            current_explanation.extend(self.candidates[substructure].coalition)
        current_explanation = list(set(current_explanation))
        if len(current_explanation) >= self.explanation_size:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            for each_node in unvisited_candidates:
                new_graph_coalition = [candidate for candidate in self.candidates 
                                       if candidate in tree_node.coalition or candidate == each_node]

                new_graph_coalition = []
                for candidate in range(len(self.candidates)): 
                    if candidate in tree_node.coalition or candidate == each_node: 
                        new_graph_coalition.append(candidate)
                new_graph_coalition = list(set(new_graph_coalition))

                new_graph_coalition = sorted(new_graph_coalition)
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Counter(new_node.coalition) == Counter(tree_node.coalition):
                    Find_same_child = True

                if Find_same_child == False:
                    tree_node.children.append(new_node)

            scores = compute_scores_exploration(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        if len(tree_node.children) == 0:
            return tree_node.P
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        if verbose:
            print(f"There are {self.num_candidates} candidates")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"[Exploration] Rollout {rollout_idx}: {len(self.state_map)} accumulative permutations.")

        explanations = []
        for _, node in self.state_map.items():
            graph_coalition = []
            for substructure in node.coalition:
                graph_coalition.extend(self.candidates[substructure].coalition)
            graph_coalition = list(set(graph_coalition))
            n = MCTSNode(graph_coalition, data=node.data, ori_graph=node.ori_graph, P=node.P)
            explanations.append(n)
            
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


def compute_scores_exploration(score_func, children: list): # list[exploration_MCTSNode]
    results = []
    for child in children: 
        if child.P == 0:
            subgraph_coalition = []
            for substructure_id in child.coalition:             
                subgraph_coalition.extend(child.candidates[substructure_id].coalition)
            score = score_func(subgraph_coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


def find_explanations(results, max_nodes=5, subgraph_building_method=None, **kwargs):
    """ return the highest reward graph node constraining to the subgraph size """
    gamma = kwargs.get('config').explainers.param.single_explanation_size
    g = results[0].ori_graph
    b = results[0].P
    data = kwargs.get('data')
    # result_node = []
    _results = [tmp for tmp in results if len(tmp.coalition) <= gamma]
    results = _results if len(_results) > 0 else results[0]
    results = sorted(results, key=lambda x: x.P, reverse=True)
    K = kwargs.get('config').explainers.param.candidate_size
    max_i = min(K, len(results))
    
    if kwargs.get('config').models.param.graph_classification:
        value_func = GnnNets_GC2value_func(kwargs.get('gnnNets'), target_class=data.y)
        score_func = reward_func(kwargs.get('config').explainers.param, value_func, subgraph_building_method=subgraph_building_method)
    else:
        value_func = GnnNets_NC2value_func(kwargs.get('gnnNets'), target_class=kwargs.get('target_class'), node_idx=kwargs.get('node_idx'))
        score_func = reward_func(kwargs.get('config').explainers.param, value_func, subgraph_building_method=subgraph_building_method)
    
    method = kwargs.get('config').explainers.param.explanation_exploration_method
    if method.lower() == 'permutation':
        def dfs(coalition: list, current, num_of_g=0):
            num_of_g = num_of_g + 1
            co = coalition # list(set(coalition))
            if len(co) > max_nodes:
                return
            if num_of_g >= 2:
                n = MCTSNode(co, data=data, ori_graph=g)
                n.P = score_func(n.coalition, data)
                results.append(n)
                # return
            for i in range(current, max_i):
                tmp = coalition.copy()
                tmp.extend(results[i].coalition)
                tmp = list(set(tmp))
                if Counter(co) == Counter(tmp):
                    continue
                dfs(tmp, i+1, num_of_g)
            pass

        dfs([], 0)
        results = sorted(results, key=lambda x: x.P, reverse=True)
    elif method.lower() == 'mcts':
        mcts_state_map = exploration_MCTS(data.x, data.edge_index, results[:max_i],
                                          score_func=score_func,
                                          n_rollout=10,
                                          explanation_size=max_nodes, 
                                          c_puct=kwargs.get('config').explainers.param.c_puct)
        results = mcts_state_map.mcts(verbose=True)
        pass
    
    result_node = results[0]
    return result_node
