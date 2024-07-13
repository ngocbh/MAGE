### Utils to transform edge masks
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

def mask_to_shape(mask, edge_index, num_top_edges):
    """Modify the mask by selecting only the num_top_edges edges with the highest mask value."""
    indices = topk_edges_unique(mask, edge_index, num_top_edges)
    unimportant_indices = [i for i in range(len(mask)) if i not in indices]
    new_mask = mask.copy()
    new_mask[unimportant_indices] = 0
    return new_mask

def topk_edges_unique(edge_mask, edge_index, num_top_edges):
    """Return the indices of the top-k edges in the mask.

    Args:
        edge_mask (Tensor): edge mask of shape (num_edges,).
        edge_index (Tensor): edge index tensor of shape (2, num_edges)
        num_top_edges (int): number of top edges to be kept
    """
    indices = (-edge_mask).argsort()
    top = np.array([], dtype="int")
    i = 0
    list_edges = np.sort(edge_index.cpu().T, axis=1)
    while len(top) < num_top_edges:
        subset = indices[num_top_edges * i : num_top_edges * (i + 1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]])
        i += 1
    return top[:num_top_edges]


def normalize_mask(x):
    if len(x) > 0 and not np.all(np.isnan(x)):
        if (np.nanmax(x) - np.nanmin(x)) == 0:
            return x
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    else:
        return x


def clean(masks):
    """Clean masks by removing NaN, inf and too small values and normalizing"""
    for i in range(len(masks)):
        if (
            (masks[i] is not None)
            and (hasattr(masks[i], "__len__"))
            and (len(masks[i]) > 0)
        ):
            masks[i] = np.nan_to_num(
                masks[i], copy=True, nan=0.0, posinf=10, neginf=-10
            )
            #print('mask without nan', masks[i])
            masks[i] = np.clip(masks[i], -10, 10)
            #print('clipped mask', masks[i])
            masks[i] = normalize_mask(masks[i])
            #print('normalized mask', masks[i])
            # masks[i] = np.where(masks[i] < 0.01, 0, masks[i])
    return masks

def transform_edge_masks(edge_masks, strategy="remove", threshold=0.1):
    if strategy == "remove":
        thresh_edge_masks = []
        for edge_mask in edge_masks:
            mask = edge_mask.copy()
            maskout = remove_hard(mask, threshold=threshold)
            thresh_edge_masks.append(maskout)
    elif strategy == "keep":
        thresh_edge_masks = []
        for edge_mask in edge_masks:
            mask = edge_mask.copy()
            masked = keep_hard(mask, threshold=threshold)
            thresh_edge_masks.append(masked)
    else:
        raise ValueError("Invalid strategy")
    return thresh_edge_masks
                                     

def keep(mask, threshold=0.1):
    mask_len = len(mask)
    split_point = int(threshold * mask_len)
    unimportant_indices = (-mask).argsort()[split_point:]
    mask[unimportant_indices] = 0
    return mask

def keep_hard(mask, threshold=0.1):
    new_mask = np.ones_like(mask)
    mask_len = len(mask)
    split_point = int(threshold * mask_len)
    unimportant_indices = (-mask).argsort()[split_point:]
    new_mask[unimportant_indices] = 0
    return new_mask

def remove(mask, threshold=0.1):
    mask_len = len(mask)
    split_point = int(threshold * mask_len)
    important_indices = (-mask).argsort()[:split_point]
    mask[important_indices] = 0
    return mask

def remove_hard(mask, threshold=0.1):
    new_mask = np.ones_like(mask)
    mask_len = len(mask)
    split_point = int(threshold * mask_len)
    important_indices = (-mask).argsort()[:split_point]
    new_mask[important_indices] = 0
    return new_mask



##### Mask properties #####

def get_sparsity(masks):
    sparsity = 0
    for i in range(len(masks)):
        sparsity += 1.0 - (masks[i] != 0).sum() / len(masks[i])
    return sparsity / len(masks)


def get_size(masks):
    size = 0
    for i in range(len(masks)):
        size += (masks[i] != 0).sum()
    return size / len(masks)


def get_entropy(masks):
    ent = 0
    k = 0
    for i in range(len(masks)):
        pos_mask = masks[i][masks[i] > 0]
        if len(pos_mask) == 0:
            continue
        ent += entropy(pos_mask)
        k += 1
    if k == 0:
        return -1
    return ent / k


def get_avg_max(masks):
    max_avg = 0
    k = 0
    for i in range(len(masks)):
        pos_mask = masks[i][masks[i] > 0]
        if len(pos_mask) == 0:
            continue
        # kde = gaussian_kde(np.array(pos_mask))
        # density = kde(pos_mask)
        # index = np.argmax(density)
        ys, xs, _ = plt.hist(pos_mask, bins=100)
        index = np.argmax(ys)
        max_avg += xs[index]
        k += 1
    if k == 0:
        return -1
    return max_avg / k


def get_mask_properties(masks):
    mask_info = {
        "mask_size": get_size(masks),
        "mask_sparsity": get_sparsity(masks),
        "mask_entropy": get_entropy(masks),
        "max_avg": get_avg_max(masks),
    }
    return mask_info
