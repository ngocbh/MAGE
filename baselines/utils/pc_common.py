

def graph_build_point_cloud(X, edge_index, node_mask: np.array):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index
