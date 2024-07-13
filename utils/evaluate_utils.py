import numpy as np
import torch
import networkx as nx
from sklearn.metrics import (
    roc_auc_score,
    adjusted_mutual_info_score,
)

from mage.utils import gnn_helpers
from mage.utils.gnn_helpers import to_networkx
from mage.utils.validation import check_random_state
from mage.utils.gnn_helpers import MaskedDataset
from torch_geometric.loader import DataLoader


def get_explanation_syn(data, edge_mask, node_list, max_nodes=None, max_edges=None):
    """Create an explanation graph from the edge_mask.
    Args:
        data (Pytorch data object): the initial graph as Data object
        edge_mask (Tensor): the explanation mask
        top_acc (bool): if True, use the top_acc as the threshold for the edge_mask
    Returns:
        G_masked (networkx graph): explanatory subgraph
    """
    ### remove self loops
    edge_index = data.edge_index.cpu().numpy()
    assert edge_index.shape[-1] == edge_mask.shape[-1]
    self_loop_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, self_loop_mask]
    edge_mask = edge_mask[self_loop_mask]

    order = (-edge_mask).argsort()
    G = nx.Graph()

    if node_list is not None:
        G.add_nodes_from(node_list)

    for i in order:
        u, v = edge_index[:, i]
        if max_edges is not None and G.number_of_edges() >= max_edges:
            break

        if max_nodes is not None and G.number_of_nodes() >= max_nodes and \
                (u not in G.nodes or v not in G.nodes):
            continue

        if edge_mask[i] == 0:
            continue

        G.add_edge(u, v)

    return G


def nx_intersection(G1, G2):
    H = nx.Graph()
    H.add_nodes_from(set(G1.nodes).intersection(G2.nodes))

    for u, v in G1.edges:
        if (u, v) in G2.edges or (v, u) in G2.edges:
            H.add_edge(u, v)

    return H


def get_edge_mask_prf1(G1, G2):
    """Compute recall, precision, and f1 score of a graph.

    Args:
        G1 (networkx graph): ground truth graph
        G2 (networkx graph): explanation graph
    """
    G1, G2 = G1.to_undirected(), G2.to_undirected()
    g_int = nx_intersection(G1, G2)
    g_int.remove_nodes_from(list(nx.isolates(g_int)))

    n_tp = g_int.number_of_edges()
    n_fp = len(G1.edges() - g_int.edges())
    n_fn = len(G2.edges() - g_int.edges())

    if n_tp == 0:
        precision, recall, f1_score = 0., 0., 0.
    else:
        precision = n_tp / (n_tp + n_fp)
        recall = n_tp / (n_tp + n_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def get_node_mask_prf1(G1, G2):
    """Compute recall, precision, and f1 score of a graph.

    Args:
        G1 (networkx graph): ground truth graph
        G2 (networkx graph): explanation graph
    """
    g1_nodes = set(G1.nodes())
    g2_nodes = set(G2.nodes())
    g_int = g1_nodes.intersection(g2_nodes)
    n_tp = len(g_int)
    n_fp = len(G1.nodes() - g_int)
    n_fn = len(G2.nodes() - g_int)

    if n_tp == 0:
        precision, recall, f1_score = 0., 0., 0.
    else:
        precision = n_tp / (n_tp + n_fp)
        recall = n_tp / (n_tp + n_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def get_ami_score(num_nodes, gt_graph, expl_graph):
    def to_cluster_assignments(num_nodes, graph):
        asm = np.zeros(num_nodes)
        for i, cpn in enumerate(nx.connected_components(graph)):
            asm[list(cpn)] = i + 1
        return asm

    gt_asm = to_cluster_assignments(num_nodes, gt_graph)
    expl_asm = to_cluster_assignments(num_nodes, expl_graph)
    ami_score = adjusted_mutual_info_score(gt_asm, expl_asm)
    return ami_score


def alpha_sampling(G, alpha):
    H = nx.Graph()
    
    for u in G.nodes:
        if np.random.rand() <= alpha:
            H.add_node(u)
    for u, v in G.edges:
        if u in H.nodes and v in H.nodes:
            H.add_edge(u, v)

    return H


def graph_minus(G1, G2):
    H = nx.Graph()
    for u in G1.nodes:
        if u not in G2.nodes:
            H.add_node(u)

    for u, v in G1.edges:
        if u in H.nodes and v in H.nodes:
            H.add_edge(u, v)
    return H


def graph_plus(G1, G2, edge_list):
    H = nx.Graph()
    for u in G1.nodes:
        H.add_node(u)
    for u in G2.nodes:
        H.add_node(u)
    
    for u, v in edge_list:
        if u in H.nodes and v in H.nodes:
            H.add_edge(u, v)

    return H


def edge_filter(data, edge_list):
    filtered_data = data.clone()
    edge_index = data.edge_index.cpu().numpy().T
    edge_mask = torch.zeros(data.edge_index.shape[-1]).to(data.x.device)
    edge_set = set(edge_list)
    for i, (u, v) in enumerate(edge_index):
        if (u, v) in edge_set or (v, u) in edge_set:
            edge_mask[i] = 1.

    filtered_data.edge_index = data.edge_index[:, edge_mask.bool()]
    return filtered_data 


def compute_payoffs(input, reward_func, masks, subgraph_building_method):
    masked_dataset = MaskedDataset(input, masks, subgraph_building_method)
    masked_dataloader = DataLoader(
        masked_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    masked_payoff_list = []
    for masked_data in masked_dataloader:
        masked_data.to(input.x.device)
        masked_payoff_list.append(reward_func(masked_data))

    masked_payoffs = torch.cat(masked_payoff_list, dim=0)
    return masked_payoffs.detach().cpu().numpy()


def fidelity_alpha_plus(predict_fn, data, expl_graph, alpha=0.5,
                        num_samples=500, subgraph_building_method='split'):
    scores = []
    graph = to_networkx(data, to_undirected=True)
    ori_prob = predict_fn(data).cpu().item()
    mask_lst = []

    for _ in range(num_samples):
        expl_graph_plus = alpha_sampling(expl_graph, alpha)
        non_expl_graph = graph_minus(graph, expl_graph_plus)
        # if not nx.is_connected(non_expl_graph):
        #     continue
        mask = torch.zeros(data.num_nodes)
        mask[torch.LongTensor(list(non_expl_graph.nodes))] = 1
        mask_lst.append(mask)

    excluded_masks = torch.vstack(mask_lst)
    excluded_probs = compute_payoffs(data, predict_fn, excluded_masks, subgraph_building_method)
    scores = ori_prob - excluded_probs

    if len(scores) != 0:
        return np.mean(scores)
    else:
        return 0


def fidelity_alpha_minus(predict_fn, data, expl_graph, alpha=0.5,
                         num_samples=500, subgraph_building_method='split'):
    scores = []
    graph = to_networkx(data, to_undirected=True)
    ori_prob = predict_fn(data).cpu().item()
    mask_lst = []

    for _ in range(num_samples):
        non_expl_graph = graph_minus(graph, expl_graph)
        non_expl_graph_minus = alpha_sampling(non_expl_graph, alpha)
        expl_graph_minus = graph_plus(non_expl_graph_minus, expl_graph, graph.edges)
        # if not nx.is_connected(expl_graph_minus):
        #     continue
        mask = torch.zeros(data.num_nodes)
        mask[torch.LongTensor(list(expl_graph_minus.nodes))] = 1
        mask_lst.append(mask)

    included_masks = torch.vstack(mask_lst)
    included_probs = compute_payoffs(data, predict_fn, included_masks, subgraph_building_method)
    scores = ori_prob - included_probs

    if len(scores) != 0:
        return np.mean(scores)
    else:
        return 0


def fidelity_alpha(predict_fn, data, expl_graph, alpha=0.5, num_samples=500, subgraph_building_method='split'):
    """
        Robust fidelity metrics
        check https://arxiv.org/pdf/2310.01820v1.pdf
    """
    # For fair comparison, make sure every method is at the same random_state

    fid_plus = fidelity_alpha_plus(predict_fn, data, expl_graph,
                                   alpha=alpha, num_samples=num_samples, 
                                   subgraph_building_method=subgraph_building_method)
    fid_minus = fidelity_alpha_minus(predict_fn, data, expl_graph,
                                     alpha=1-alpha, num_samples=num_samples,
                                     subgraph_building_method=subgraph_building_method)
    fid_delta = fid_plus - fid_minus
    return fid_delta, fid_plus, fid_minus


def compute_explanation_stats(data, gt_edge_mask=None, num_motifs=1, edge_mask=None,
                              node_list=None, max_nodes=None, get_max=False,
                              predict_fn=None, subgraph_building_method='split'):
    # evaluation should be consistent
    # subgraph_building_method should depend on the data rather than method
    # for prediction tasks focusing on the graph structure, split is good 
    # for tasks that highly depends on node features, zero_filling is more appropriate
    # all method should use the same subgraph_building_method for evaluation

    assert (node_list is not None) or (edge_mask is not None)
    assert (max_nodes is not None)

    stats, info = {}, {}
    max_nodes = max_nodes
    expl_graph = get_explanation_syn(data, edge_mask, node_list, max_nodes=max_nodes)
    # robust fidelity metrics
    fid_delta, fid_plus, fid_minus = fidelity_alpha(predict_fn, data, expl_graph,
                                                    alpha=.7, num_samples=2000,
                                                    subgraph_building_method=subgraph_building_method)

    # in case alpha = 1.0, it recovers original fidelity metrics used in SubgraphX and GStarx
    _, org_fid, org_fid_inv = fidelity_alpha(predict_fn, data, expl_graph,
                                             alpha=1, num_samples=1,
                                             subgraph_building_method=subgraph_building_method)
    stats['fid_delta'] = fid_delta
    stats['fid_plus'] = fid_plus
    stats['fid_minus'] = fid_minus
    stats['org_fid'] = org_fid
    stats['org_fid_inv'] = org_fid_inv
    stats['org_fid_delta'] = org_fid - org_fid_inv
    info['expl_graph'] = expl_graph

    if gt_edge_mask is not None:
        gt_graph = get_explanation_syn(data, edge_mask=gt_edge_mask, node_list=None)
        if num_motifs == 1 and get_max:
            best_p, best_r, best_f1 = 0., 0., 0.
            best_pn, best_rn, best_f1n = 0., 0., 0.

            cpns = list(nx.connected_components(gt_graph))
            for cpn in cpns:
                p, r, f = get_edge_mask_prf1(gt_graph.subgraph(list(cpn)), expl_graph)
                pn, rn, fn = get_node_mask_prf1(gt_graph, expl_graph)
                if f > best_f1:
                    best_p, best_r, best_f1 = p, r, f
                    best_pn, best_rn, best_f1n = pn, rn, fn
        else:
            best_p, best_r, best_f1 = get_edge_mask_prf1(gt_graph, expl_graph)
            best_pn, best_rn, best_f1n = get_node_mask_prf1(gt_graph, expl_graph)

        ami = get_ami_score(data.num_nodes, gt_graph, expl_graph)

        stats['precision'] = best_p
        stats['recall'] = best_r
        stats['f1_score'] = best_f1
        stats['node_precision'] = best_pn
        stats['node_recall'] = best_rn
        stats['node_f1_score'] = best_f1n
        stats['ami_score'] = ami

        try:
            stats['auc'] = roc_auc_score(gt_edge_mask, edge_mask)
        except:
            stats['auc'] = 0.

        info['gt_graph'] = gt_graph

    return stats, info
