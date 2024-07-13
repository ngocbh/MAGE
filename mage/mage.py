import networkx as nx
import numpy as np
import time
import torch

from functools import partial
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils.subgraph import k_hop_subgraph, get_num_hops

from mage.indices import shapley_taylor_indices, shapley_interaction_indices
from mage.utils.validation import check_random_state, check_graph_input
from mage.utils.gnn_helpers import get_reward_func_for_gnn_gc, to_networkx, normalize_reward_func
from mage.utils.gnn_helpers import subgraph_connected_components
from mage.maskers import MaskedDataset
from mage.motif_search import _find_motifs_with_connectivity, _find_motifs


cc_time = 0
fn_time = 0


def communication_restricted_fn_wrapper(T, fn, G: nx.Graph, cpn_dict, restricted=True):
    if T in cpn_dict:
        return 0

    if restricted:
        cpns = subgraph_connected_components(G.adj, T)
    else:
        cpns = [T]

    ret = []
    for C in cpns:
        fn_c = fn(frozenset(C))
        ret.append(fn_c)

    cpn_dict[T] = ret
    return 0


def dry_fn_wrapper(T: frozenset, num_nodes: int, mask_dict: dict, mask_list: list):
    if T not in mask_dict:
        mask_list.append(T)
        idx = len(mask_list) - 1
        mask_dict[T] = len(mask_list) - 1
        return idx
    else:
        return mask_dict[T]


def wet_fn_wrapper(T: frozenset, cpn_dict: dict, masked_payoffs: list):
    if T not in cpn_dict:
        raise ValueError(f"{T} is not in cpn_dict")
    else:
        cpn_ids = cpn_dict[T]
        ret = 0
        for idx in cpn_ids:
            ret += masked_payoffs[idx]

    return ret


class Mage():
    def __init__(self, model, masker,
                 payoff_type="norm_prob",
                 random_state=None, device='cpu'):
        self.model = model
        self.rng = check_random_state(random_state)
        self.device = device
        self.payoff_type = payoff_type
        self.masker = masker

    def get_reward_func(self, target_class, input_masker):
        reward_func = get_reward_func_for_gnn_gc(self.model, target_class, self.payoff_type)
        if 'norm' in self.payoff_type:
            # important, always use norm payoff_type
            emp_data = input_masker([]).to(self.device)
            f_emp = reward_func(emp_data).item()
            return normalize_reward_func(reward_func, f_emp)
        else:
            # dont use this
            return reward_func

    def explain(
        self,
        input,
        num_motifs,
        beta,
        tau=0.5,
        epsilon=0.1,
        omega=0.5,
        ord=2,
        method='myt',
        num_samples=1000,
        num_segments=-1,
        target_class='max',
        connectivity='viky',
    ):
        start_time = time.time()
        # get target_class
        logit = self.model(input)
        if target_class == 'max':
            target_class = logit.argmax(-1).item()
        elif not isinstance(target_class, int):
            raise ValueError('unrecognized target class')

        self.model.eval()

        graph = self.masker.build_rag(input, num_segments=num_segments)
        input_masker = partial(self.masker, input=input)

        reward_func = self.get_reward_func(target_class, input_masker=input_masker)

        calc_indices_ts = time.time()
        # Calc interaction_indices
        self.num_queries = 0
        indices = self.interaction_indices(
            graph,
            reward_func=reward_func,
            input_masker=input_masker,
            method=method,
            ord=ord,
            num_samples=num_samples
        )
        calc_indices_te = time.time()
        
        # with open('indices.npy', 'wb') as f:
        #     np.save(f, np.array(indices))
        # with open('indices.npy', 'rb') as f:
        #     indices = np.load(f)
        # print(indices)
        # raise ValueError

        normalized_indices = self.normalize_indices(indices, tau, epsilon)

        motifs = _find_motifs_with_connectivity(
            normalized_indices,
            G=graph,
            num_motifs=num_motifs,
            beta=beta,
            omega=omega,
            ord=ord,
            max_iterations=5,
            connectivity=connectivity,
        )
        calc_motifs_te = time.time()

        end_time = time.time()
        related_preds = {}
        related_preds['running_time'] = end_time - start_time
        related_preds['calc_indices_time'] = calc_indices_te - calc_indices_ts
        related_preds['find_motifs_time'] = calc_motifs_te - calc_indices_te
        related_preds['num_queries'] = self.num_queries
        info = {
            "related_preds": related_preds,
            "indices": indices,
            "normalized_indices": normalized_indices,
            "adj": nx.adjacency_matrix(graph).todense(),
        }

        return motifs, info

    def normalize_indices(self, indices, tau, epsilon=0.1):
        # compute weighted interactions
        indices_pos = np.maximum(0, indices)
        indices_neg = np.minimum(0, indices)

        # cancel out the noise of dummy nodes' contributions
        max_val_pos = np.max(np.abs(indices_pos))
        indices_pos[np.where(np.abs(indices_pos) <= max_val_pos * epsilon)] = 0

        max_val_neg = np.max(np.abs(indices_neg))
        indices_neg[np.where(np.abs(indices_neg) <= max_val_neg * epsilon)] = 0

        # Compute normalized interaction matrix
        normalized_indices = tau * indices_pos + (1 - tau) * indices_neg
        return normalized_indices

    def interaction_indices(
        self,
        graph,
        reward_func,
        input_masker,
        method="myt",
        ord=2,
        num_samples=100,
    ):
        num_nodes = graph.number_of_nodes()

        # dry run
        print('begin dry run...')
        mask_dict = {}
        mask_list = []
        cpn_dict = {}

        par_dry_func = partial(dry_fn_wrapper, num_nodes=num_nodes, mask_dict=mask_dict, mask_list=mask_list)

        restricted = (method in ['myi', 'myt'])
            
        dry_func = partial(communication_restricted_fn_wrapper,
                           fn=par_dry_func,
                           G=graph,
                           cpn_dict=cpn_dict,
                           restricted=restricted)

        # use a fixed seed to make sure dry and wet payoff function will explore the same coalition set
        seed = self.rng.randint(1e9+7)
        t1 = time.time()
        if method in ['sht', 'myt']:
            shapley_taylor_indices(num_players=num_nodes,
                                   fn=dry_func,
                                   ord=ord,
                                   num_samples=num_samples,
                                   random_state=seed,
                                   return_indices=False)
        else:
            shapley_interaction_indices(num_players=num_nodes,
                                        fn=dry_func,
                                        ord=ord,
                                        num_samples=num_samples,
                                        random_state=seed,
                                        return_indices=False)

        t2 = time.time()

        print('computing payoffs...')
        masked_payoffs = self.compute_payoffs(reward_func, input_masker, mask_list)

        t3 = time.time()

        wet_func = partial(wet_fn_wrapper, cpn_dict=cpn_dict, masked_payoffs=masked_payoffs)

        print('aggregating results...')
        if method in ['sht', 'myt']:
            indices = shapley_taylor_indices(num_players=num_nodes,
                                             fn=wet_func,
                                             ord=ord,
                                             num_samples=num_samples,
                                             random_state=seed)
        else:
            indices = shapley_interaction_indices(num_players=num_nodes,
                                                  fn=wet_func,
                                                  ord=ord,
                                                  num_samples=num_samples,
                                                  random_state=seed)

        t4 = time.time()
        return indices

    def compute_payoffs(self, reward_func, masker_func, masks):
        masked_dataset = MaskedDataset(masks, masker_func)
        masked_dataloader = DataLoader(
            masked_dataset, batch_size=256, shuffle=False, num_workers=0
        )

        masked_payoff_list = []
        print('num_queries: ', len(masked_dataset))
        self.num_queries += len(masked_dataset)
        for masked_data in masked_dataloader:
            masked_data.to(self.device)
            masked_payoff_list.append(reward_func(masked_data))

        masked_payoffs = torch.cat(masked_payoff_list, dim=0)
        return masked_payoffs.detach().cpu().numpy()


class MageNC(Mage):

    def get_reward_func(self, input, target_class):
        pass

    def get_k_hop_subgraph(self, node_idx, num_hops, input):
        subset, edge_index, mapping, subgraph_edge_mask = k_hop_subgraph(node_idx, num_hops, input.edge_index)
        pass

    def explain(self, node_idx, input, target_class, num_hops=None, *args, **kwargs):
        if num_hops is None:
            num_hops = get_num_hops(self.model)

        input = self.get_k_hop_subgraph(node_idx, input, num_hops)
        raise ValueError
