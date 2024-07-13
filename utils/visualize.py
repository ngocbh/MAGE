import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import itertools

from rdkit import Chem
from typing import Union, List
from textwrap import wrap
from torch_geometric.datasets import MoleculeNet
from mage.utils.validation import check_mask_input


def normalize_zero_one(im, eps=1e-7):
    m1 = im.min()
    m2 = im.max()
    return (im - m1) / (m2 - m1 + eps)


def get_group_attribution(indices, mask):
    n = indices.shape[0]
    mask = check_mask_input(mask, size=n)
    if len(indices.shape) == 1:
        return np.sum(mask * indices)
    elif len(indices.shape) == 2:
        return np.sum(mask.T @ indices @ mask + np.diag(indices).T @ mask) / 2


class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, figname, nodelist=None, expl_graph=None, title_sentence=None, **kwargs):
        if 'indices' in kwargs:
            indices = kwargs.get('indices')
            motif_weights = []
            for motif in nodelist:
                gr_attr = get_group_attribution(indices, motif)
                motif_weights.append(gr_attr)
        else:
            motif_weights = None
                
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp', 'ba_house_grid', 
                                         'ba_house_and_grid', 'ba_house_or_grid', 'spmotif']:
            if 'label_on' in kwargs and kwargs.get('label_on'):
                label_on = True
            else:
                label_on = False
            edgelist = list(expl_graph.edges) if expl_graph is not None else None
            nodelist = [list(cpn) for cpn in nx.connected_components(expl_graph)] if expl_graph is not None else nodelist
            self.plot_ba2motifs(graph, nodelist=nodelist, edgelist=edgelist, motif_weights=motif_weights, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['mutag', 'mutag0', 'benzene'] + list(MoleculeNet.names.keys()):
            edgelist = list(expl_graph.edges) if expl_graph is not None else None
            nodelist = [list(cpn) for cpn in nx.connected_components(expl_graph)] if expl_graph is not None else nodelist
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, edgelist=edgelist, motif_weights=motif_weights, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['mnist75sp']:
            data = kwargs.get('data')
            indices = kwargs.get('indices') if 'indices' in kwargs else None
            self.plot_image(graph, nodelist=nodelist, data=data, indices=indices,
                            figname=figname, title_sentence=title_sentence)
        elif self.dataset_name.lower() in ['modelnet40', 'modelnet10']:
            data = kwargs.get('data')
            indices = kwargs.get('indices') if 'indices' in kwargs else None
            self.plot_point_cloud(graph, nodelist=nodelist, data=data, indices=indices,
                            figname=figname, title_sentence=title_sentence)
        else:
            raise NotImplementedError


    def plot_point_cloud(self, graph, data, figname, nodelist=None, indices=None, title_sentence=None):
        raise ValueError

    def plot_image(self, graph, data, figname, nodelist=None, indices=None, title_sentence=None):
        # mean and std computed for superpixel features
        # features are 3 pixel intensities and 2 coordinates (x,y)
        # 3 pixel intensities because we replicate mean pixel value 3 times to test on colored MNIST images
        mn = torch.tensor([0.11225057, 0.11225057, 0.11225057, 0.44206527, 0.43950436]).view(1, 1, -1)
        sd = torch.tensor([0.2721889,  0.2721889,  0.2721889,  0.2987583,  0.30080357]).view(1, 1, -1)
        # print(graph)
        sp_order = data.sp_order.cpu().numpy()
        sp_intensity = data.sp_intensity.cpu().numpy()
        sp_coord = data.sp_coord
        superpixels = data.superpixels.cpu().numpy()
        img_size = data.img_size.item()

        fig, ax = plt.subplots()
        img = np.zeros((img_size, img_size))

        if nodelist is None:
            for j, (sp_intens, sp_index) in enumerate(zip(sp_intensity, sp_order)):
                mask = (superpixels == sp_index)
                x = (sp_intens - mn[0, 0, 0].item()) / sd[0, 0, 0].item()
                img[mask] = x

            ax.imshow(img, cmap='gray')
        elif indices is None:
            flattened = list(itertools.chain(*nodelist))
            for j, (sp_intens, sp_index) in enumerate(zip(sp_intensity, sp_order)):
                mask = (superpixels == sp_index)
                x = (1 - mn[0, 0, 0].item()) / sd[0, 0, 0].item() if j in flattened else 0.
                img[mask] = x

            ax.imshow(normalize_zero_one(img))
        else:
            P = np.zeros_like(indices)
            for motif in nodelist:
                P[np.ix_(motif, motif)] = indices[np.ix_(motif, motif)]
            scores = np.triu(P).sum(axis=-1)
            # scores = np.triu(indices).sum(axis=-1)
            # scores = np.diag(indices)

            flattened = list(itertools.chain(*nodelist))
            for j, (sp_intens, sp_index) in enumerate(zip(sp_intensity, sp_order)):
                mask = (superpixels == sp_index)
                x = scores[j]
                img[mask] = x

            ax.imshow(normalize_zero_one(img))


        plt.axis('off')
        # if title_sentence is not None:
        #     plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname, dpi=200, bbox_inches='tight')
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_subgraph(
        self,
        graph,
        nodelist,
        colors: Union[None, str, List[str]] = "#FFA500",
        labels=None,
        motif_weights=None,
        edge_color="gray",
        edgelist=None,
        subgraph_colors=None,
        title_sentence=None,
        figname=None,
    ):
        # if subgraph_colors is None:
        #     subgraph_colors = ['green', 'blue', 'red', 'orange', 'pink', 'brown']
        motif_colors = ['red', 'lime']
        if motif_weights is None:
            motif_weights = np.ones(len(nodelist))

        if edgelist is None:
            edgelist = [
                (n_frm, n_to)
                for (n_frm, n_to) in graph.edges()
                if any((n_frm in c and n_to in c) for c in nodelist)
            ]

        pos = nx.kamada_kawai_layout(graph)


        for i, cluster in enumerate(nodelist):
            pos_nodelist = {k: v for k, v in pos.items() if k in cluster}
            nx.draw_networkx_nodes(
                graph,
                pos_nodelist,
                nodelist=cluster,
                node_color=motif_colors[int(motif_weights[i] > 0)],
                # alpha=0.99 * (np.abs(motif_weights[i])/np.max(np.abs(motif_weights))),
                alpha=0.5,
                node_shape="o",
                node_size=800,
            )

        nx.draw_networkx_nodes(
            graph, pos, nodelist=list(graph.nodes()), node_color=colors, node_size=300
        )

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)


        for i, cluster in enumerate(nodelist):
            pos_nodelist = {k: v for k, v in pos.items() if k in cluster}
            edgelist = [
                (n_frm, n_to)
                for (n_frm, n_to) in graph.edges()
                if n_frm in cluster and n_to in cluster
            ]
            nx.draw_networkx_edges(
                graph.subgraph(cluster),
                pos=pos_nodelist,
                edgelist=edgelist,
                width=10,
                edge_color=motif_colors[int(motif_weights[i] > 0)],
                alpha=0.5,
                # alpha=0.99 * (np.abs(motif_weights[i])/np.max(np.abs(motif_weights))),
                arrows=False,
            )

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels, font_weight='bold')

        plt.axis("off")
        # if title_sentence is not None:
        #     plt.title("\n".join(wrap(title_sentence, width=60)))
        if figname is not None:
            plt.savefig(figname, dpi=200, bbox_inches='tight', transparent=True)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_subgraph_with_nodes(self,
                                 graph,
                                 nodelist,
                                 node_idx,
                                 colors='#FFA500',
                                 labels=None,
                                 edge_color='gray',
                                 edgelist=None,
                                 subgraph_edge_color='black',
                                 title_sentence=None,
                                 figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph)  # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(
        self, graph, nodelist=None, edgelist=None, label_on=False, motif_weights=None, title_sentence=None, figname=None
    ):
        # node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        if label_on:
            node_labels = {k: k for k, u in enumerate(graph.nodes)}
        else:
            node_labels = None
        return self.plot_subgraph(
            graph,
            nodelist,
            edgelist=edgelist,
            motif_weights=motif_weights,
            labels=node_labels,
            title_sentence=title_sentence,
            figname=figname,
        )

    def plot_molecule(self,
                      graph,
                      nodelist,
                      x,
                      motif_weights=None,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        # collect the text information and node color
        if self.dataset_name in ['mutag', 'mutag0', 'benzene']:
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br',
                         7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_labels = {k: k for k, u in enumerate(graph.nodes)}
            node_color = ['#E49D1C', '#4970C6', '#29A329', '#FF5357',  'brown',  'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in MoleculeNet.names.keys():
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            # node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
            #                for k, v in element_idxs.items()}
            node_labels = {k: k for k, u in enumerate(graph.nodes)}
            node_color = ['#E49D1C', '#4970C6', '#29A329', '#FF5357',  'brown',  'darkslategray', '#F0EA00']
            # node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist,
                           colors=colors,
                           labels=node_labels,
                           edgelist=edgelist,
                           motif_weights=motif_weights,
                           edge_color='gray',
                           title_sentence=title_sentence,
                           figname=figname)

    def plot_sentence(self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist[0]}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist[0],
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
            if edgelist is None:
                edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                            if n_frm in nodelist[0] and n_to in nodelist[0]]
                nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='yellow', arrows=True)

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey')
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if title_sentence is not None:
            string = '\n'.join(wrap(' '.join(words), width=50))
            string += '\n'.join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_bashapes(self,
                      graph,
                      nodelist,
                      y,
                      node_idx,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph,
                                      nodelist,
                                      node_idx,
                                      colors,
                                      edgelist=edgelist,
                                      title_sentence=title_sentence,
                                      figname=figname,
                                      subgraph_edge_color='black')


def heatmap(X, filepath=None):
    pass


def show_rag(labels, rag, image, border_color='black', edge_width=1.5,
             edge_cmap='magma', img_cmap='bone', in_place=True, ax=None, node_labels=False):
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from skimage import measure, segmentation, util, color

    if not in_place:
        rag = rag.copy()

    if ax is None:
        fig, ax = plt.subplots()
    out = util.img_as_float(image, force_copy=True)

    if img_cmap is None:
        if image.ndim < 3 or image.shape[2] not in [3, 4]:
            msg = 'If colormap is `None`, an RGB or RGBA image should be given'
            raise ValueError(msg)
        # Ignore the alpha channel
        out = image[:, :, :3]
    else:
        img_cmap = plt.get_cmap(img_cmap)
        out = color.rgb2gray(image)
        # Ignore the alpha channel
        out = img_cmap(out)[:, :, :3]

    edge_cmap = plt.get_cmap(edge_cmap)

    # Handling the case where one node has multiple labels
    # offset is 1 so that regionprops does not ignore 0
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for n, d in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1

    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)

    for (n, data), region in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))

    cc = colors.ColorConverter()
    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)

    ax.imshow(out)

    # Defining the end points of the edges
    # The tuple[::-1] syntax reverses a tuple as matplotlib uses (x,y)
    # convention while skimage uses (row, column)
    lines = [[rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
              for (n1, n2) in rag.edges()]

    if node_labels:
        for n, d in rag.nodes(data=True):
            x, y = rag.nodes[n]['centroid'][::-1]
            ax.text(x, y, d['labels'][0], color='blue', fontsize=13)

    lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    # edge_weights = [d['weight'] for x, y, d in rag.edges(data=True)]
    edge_weights = [1 for x, y, d in rag.edges(data=True)]
    lc.set_array(np.array(edge_weights))
    ax.add_collection(lc)
    return lc


def show_explanation(input, motifs, attribution, segments, segment_indices, rag,
                     border_color='black', edge_width=1.5,
                     edge_cmap='magma', img_cmap='bone',
                     ax=None, alpha=0.4, node_labels=False, show_edges=True):
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from skimage import measure, segmentation, util, color

    rag = rag.copy()
    image = input.copy()

    expl_graph = nx.Graph()
    expl_graph.add_nodes_from([u for motif in motifs for u in motif])
    for n1, n2 in rag.edges():
        for motif in motifs:
            if n1 in motif and n2 in motif:
                expl_graph.add_edge(n1, n2)

    if ax is None:
        fig, ax = plt.subplots()
    out = util.img_as_float(image, force_copy=True)
    
    if img_cmap is None:
        if image.ndim < 3 or image.shape[2] not in [3, 4]:
            msg = 'If colormap is `None`, an RGB or RGBA image should be given'
            raise ValueError(msg)
        # Ignore the alpha channel
        # out = image[:, :, :3]
    else:
        img_cmap = plt.get_cmap(img_cmap)
        # out = image
        # out = color.rgb2gray(image)
        # Ignore the alpha channel
        out = (0.2989 * out[:, :, 0] + 0.5870 * out[:, :, 1] + 0.1140 * out[:, :, 2])  # rgb to gray
        out = img_cmap(out)[:, :, :3]

    sv = np.zeros_like(out[:, :, 0])
    # print(expl_graph.nodes)
    for n in expl_graph.nodes():
        mask = (segments == segment_indices[n])
        sv[mask] = attribution[n]

    edge_cmap = plt.get_cmap(edge_cmap)

    # Handling the case where one node has multiple labels
    # offset is 1 so that regionprops does not ignore 0
    offset = 1
    map_array = np.arange(segments.max() + 1)
    for n, d in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1

    rag_labels = map_array[segments]
    regions = measure.regionprops(rag_labels)

    for (n, data), region in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))

    cc = colors.ColorConverter()
    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)
    
    max_val = np.nanpercentile(np.abs(attribution), 99.9)
    ax.imshow(out, alpha=alpha, cmap=plt.get_cmap('gray'))
    im = ax.imshow(sv, vmin=-max_val, vmax=max_val, alpha=alpha*1.5)
    # Defining the end points of the edges
    # The tuple[::-1] syntax reverses a tuple as matplotlib uses (x,y)
    # convention while skimage uses (row, column)
    lines = [[rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
              for (n1, n2) in expl_graph.edges()]

    if node_labels:
        for n, d in rag.nodes(data=True):
            if n in expl_graph.nodes():
                x, y = rag.nodes[n]['centroid'][::-1]
                ax.text(x, y, d['labels'][0], color='blue', fontsize=13)
    if show_edges:
        lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
        # edge_weights = [d['weight'] for x, y, d in rag.edges(data=True)]
        #edge_weights = [1 for x, y, d in rag.edges(data=True)]
        # lc.set_array(np.array(attribution))
        ax.add_collection(lc)
    ax.axis('off')

    return im


def plot_point_cloud_segments(x, y, z, segments=None, filename=None):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    img = ax.scatter(x, y, z, c=segments, cmap='hsv', s=30)

    ax.view_init(45, -120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_axis_off()
    if filename != None:
        plt.savefig(filename, dpi=200, bbox_inches='tight')


def plot_point_cloud_rag(x, y, z, segments, graph, filename=None):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pos = np.array([graph.nodes[u]['pos'] for u in graph.nodes])
    sx, sy, sz = pos.T

    ax.scatter(x, y, z, c=segments, cmap='hsv', alpha=0.03, s=30)
    ax.scatter(sx, sy, sz, c=list(range(len(sx))), cmap='hsv', s=30)
    for u, v in graph.edges:
        pu, pv = graph.nodes[u]['pos'], graph.nodes[v]['pos']
        line = np.vstack([pu, pv])
        ax.plot(*line.T)

    ax.view_init(45, -120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()

    if filename != None:
        plt.savefig(filename, dpi=200, bbox_inches='tight')


def plot_point_cloud_explanation(pos, segments, graph, scores, filename=None):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.colors import TwoSlopeNorm

    plt.close('all')
    colormap = cm.bwr
    # norm = Normalize(vmin=min(heights), vmax=max(heights), clip=True)
    norm = TwoSlopeNorm(vcenter=0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # spos = np.array([graph.nodes[u]['pos'] for u in graph.nodes])
    # sx, sy, sz = spos.T

    point_scores = scores[segments]
    is_null = np.abs(point_scores) < 1e-5
    ax.scatter(*pos[is_null].T, alpha=0.05, color='gray', s=30)
    ax.scatter(*pos[~is_null].T, c=colormap(norm(point_scores[~is_null])), s=30)
    # for u, v in graph.edges:
    #     pu, pv = graph.nodes[u]['pos'], graph.nodes[v]['pos']
    #     line = np.vstack([pu, pv])
    #     ax.plot(*line.T)

    ax.view_init(45, -120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()

    if filename != None:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        
