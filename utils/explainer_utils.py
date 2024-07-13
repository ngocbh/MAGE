import torch
import networkx as nx

from mage.utils.gnn_helpers import to_networkx


def choose_explainer_param(data, dataset):
    motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()

    ## choose number of clusters
    subgraph_data = data.clone()
    subgraph_data.edge_index = data.edge_index[:, motif_edge_mask.astype(bool)]
    graph = to_networkx(subgraph_data, to_undirected=True)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    num_clusters = nx.number_connected_components(graph)
    beta = graph.number_of_nodes()
    return beta, num_clusters

def explanation_filter(dataset_name, dataset, test_indices):
    """
        Make sure to explain the suitable examples
    """
    if dataset_name in ['graph_sst2', 'graph_sst5', 'twitter']:
        # evaluate on the first 200 sentences
        return test_indices[:200]
    if dataset_name in ['mnist75sp']:
        # evaluate on the first 200 images
        return test_indices[:200]
    if dataset_name in ['spmotif']:
        # remove too big graph
        filtered_test_indices = []
        for i, data in enumerate(dataset[test_indices]):
            if data.num_nodes < 100:
                filtered_test_indices.append(test_indices[i])
        return filtered_test_indices
    if dataset_name in ['ba_house_and_grid', 'ba_house_or_grid']:
        filtered_test_indices = []
        for i, data in enumerate(dataset[test_indices]):
            if data.y.item() == 1:
                filtered_test_indices.append(test_indices[i])
        return filtered_test_indices
    if dataset_name == 'ba_house_grid':
        filtered_test_indices = []
        for i, data in enumerate(dataset[test_indices]):
            if data.y.item() > 0:
                filtered_test_indices.append(test_indices[i])
        # return filtered_test_indices
        return test_indices
    elif dataset_name == 'mutag0':
        # only explain mutagenetic example (label 0) for which ground_truth explanation exists
        filtered_test_indices = []
        for i, data in enumerate(dataset[test_indices]):
            if data.y.item() == 0:
                filtered_test_indices.append(test_indices[i])
        return filtered_test_indices
    elif dataset_name == 'benzene':
        filtered_test_indices = []
        for i, data in enumerate(dataset[test_indices]):
            if data.y.item() == 1:
                filtered_test_indices.append(test_indices[i])
        return filtered_test_indices
    else:
        return test_indices
