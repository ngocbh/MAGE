import copy
import io
import math
import os
import pickle
from collections import OrderedDict
from enum import Enum
from pathlib import Path

from torch_geometric.utils import subgraph
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from torch.nn import Linear as Lin
from torch.nn import ModuleList
from torch.nn import functional as F
from torch_geometric.nn import ARMAConv, BatchNorm, MessagePassing

n_class_dict = {'MutagNet': 2, 'Tox21Net': 2, 'Reddit5kNet': 5, 'VGNet': 5, 'BA2MotifNet': 2, 'BA3MotifNet': 3,
                'TR3MotifNet': 3, 'MNISTNet': 10}
vis_dict = {'MutagNet': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
            'Tox21Net': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
            'BA3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 3},
            'TR3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 5},
            'GraphSST2Net': {'node_size': 400, 'linewidths': 1, 'font_size': 12, 'width': 3},
            'MNISTNet': {'node_size': 100, 'linewidths': 1, 'font_size': 10, 'width': 2},
            'defult': {'node_size': 200, 'linewidths': 1, 'font_size': 10, 'width': 2}}
chem_graph_label_dict = {
    'MutagNet': {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K',
                 12: 'Li', 13: 'Ca'},
    'Tox21Net': {0: 'O', 1: 'C', 2: 'N', 3: 'F', 4: 'Cl', 5: 'S', 6: 'Br', 7: 'Si', 8: 'Na', 9: 'I', 10: 'Hg', 11: 'B',
                 12: 'K', 13: 'P', 14: 'Au', 15: 'Cr', 16: 'Sn', 17: 'Ca', 18: 'Cd', 19: 'Zn', 20: 'V', 21: 'As',
                 22: 'Li', 23: 'Cu', 24: 'Co', 25: 'Ag', 26: 'Se', 27: 'Pt', 28: 'Al', 29: 'Bi', 30: 'Sb', 31: 'Ba',
                 32: 'Fe', 33: 'H', 34: 'Ti', 35: 'Tl', 36: 'Sr', 37: 'In', 38: 'Dy', 39: 'Ni', 40: 'Be', 41: 'Mg',
                 42: 'Nd', 43: 'Pd', 44: 'Mn', 45: 'Zr', 46: 'Pb', 47: 'Yb', 48: 'Mo', 49: 'Ge', 50: 'Ru', 51: 'Eu',
                 52: 'Sc'}}
rec_color = ['cyan', 'mediumblue', 'deeppink', 'darkorange', 'gold', 'chartreuse', 'lightcoral', 'darkviolet', 'teal',
             'lightgrey', ]


def sentence_layout(sentence, length):
    num_token = len(sentence)
    pos = {}
    height = []
    width = []

    right_margin = len(sentence[-1]) * 0.05
    gap = (length - right_margin) / (num_token - 1)
    start = 0
    for i in range(num_token):
        pos[i] = np.array([start + gap * i, gap / 5 * pow(-1, i)])
        width.append(len(sentence[i]) * 0.04)
        height.append(gap / 3)
    return pos, np.array(width), np.array(height)


def e_map_mutag(bond_type, reverse=False):
    from rdkit import Chem
    if not reverse:
        if bond_type == Chem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.BondType.AROMATIC:
            return 2
        elif bond_type == Chem.BondType.TRIPLE:
            return 3
        else:
            raise Exception("No bond type found")

    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.AROMATIC
    elif bond_type == 3:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")


class x_map_mutag(Enum):
    C = 0
    O = 1
    Cl = 2
    H = 3
    N = 4
    F = 5
    Br = 6
    S = 7
    P = 8
    I = 9
    Na = 10
    K = 11
    Li = 12
    Ca = 13


def graph_to_mol(X, edge_index, edge_attr):
    from rdkit import Chem
    mol = Chem.RWMol()
    X = [Chem.Atom(x_map_mutag(x.index(1)).name) for x in X]

    E = edge_index
    for x in X:
        mol.AddAtom(x)
    for (u, v), attr in zip(E, edge_attr):
        attr = e_map_mutag(attr.index(1), reverse=True)

        if mol.GetBondBetweenAtoms(u, v):
            continue
        mol.AddBond(u, v, attr)
    return mol



class Explainer(object):
    def __init__(self, device, gnn_model):
        self.device = device
        self.model = gnn_model
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

        self.last_result = None
        self.vis_dict = None

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different g attribution methods
        :param graph: target g instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def __relabel__(self, g, edge_index):
        # get nodes of subgraph and its corresponding batch index
        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]

        # get the starting and ending nodes of the edge
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # reset the index of nodes. If nodes do not belong to the subgraph, assign -1
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)

        # update node indexes of each edge
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos

    def __reparameterize__(self, log_alpha, beta=0.1, training=True):
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def pack_explanatory_subgraph(self, top_ratio=0.2, graph=None, imp=None, relabel=True):
        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, 'length mismatch'

        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = g.x
        if relabel:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = self.__relabel__(
                exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph

    def evaluate_recall(self, topk=10):

        graph, imp = self.last_result
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        return float(values.sum()) / float(graph.ground_truth_mask.sum())

    def evaluate_acc(self, top_ratio_list, graph=None, imp=None):
        if graph is None:
            assert self.last_result is not None
            graph, imp = self.last_result
        acc = np.array([[]])
        prob = np.array([[]])
        y = graph.y
        for idx, top_ratio in enumerate(top_ratio_list):
            if top_ratio == 1.0:
                self.model(graph)
            else:
                exp_subgraph = self.pack_explanatory_subgraph(top_ratio, graph=graph, imp=imp)
                self.model(exp_subgraph)
            res_acc = (y == self.model.readout.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
            res_prob = self.model.readout[0, y].detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res_acc], axis=1)
            prob = np.concatenate([prob, res_prob], axis=1)
        return acc, prob

    def visualize(self, graph=None, edge_imp=None, counter_edge_index=None, vis_ratio=0.2, save=False, name=None):
        if graph is None:
            assert self.last_result is not None
            graph, edge_imp = self.last_result

        topk = max(int(vis_ratio * graph.num_edges), 1)
        idx = np.argsort(-edge_imp)[:topk]
        G = nx.DiGraph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(list(graph.edge_index.cpu().numpy().T))

        if counter_edge_index is not None:
            G.add_edges_from(list(counter_edge_index.cpu().numpy().T))
        if self.vis_dict is None:
            self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict['defult']

        folder = Path(r'image/%s' % self.model_name)
        if save and not os.path.exists(folder):
            os.makedirs(folder)

        edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
        edge_pos_mask[idx] = True
        vmax = sum(edge_pos_mask)
        node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
        node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
        node_pos_mask[node_pos_idx] = True
        node_neg_mask[node_neg_idx] = True

        if self.model_name == "GraphSST2Net":
            plt.figure(figsize=(10, 4), dpi=100)
            ax = plt.gca()
            node_imp = np.zeros(graph.num_nodes)
            row, col = graph.edge_index[:, edge_pos_mask].cpu().numpy()
            node_imp[row] += edge_imp[edge_pos_mask]
            node_imp[col] += edge_imp[edge_pos_mask]
            node_alpha = node_imp / max(node_imp)
            pos, width, height = sentence_layout(graph.sentence_tokens[0], length=2)

            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index.cpu().numpy().T), edge_color='whitesmoke',
                                   width=self.vis_dict['width'], arrows=True, connectionstyle="arc3,rad=0.2"
                                   # <-- THIS IS IT
                                   )
            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]), width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('Greys'), edge_vmin=-vmax, edge_vmax=vmax, arrows=True,
                                   connectionstyle="arc3,rad=0.2")

            for i in node_pos_idx:
                patch = Rectangle(xy=(pos[i][0] - width[i] / 2, pos[i][1] - height[i] / 2), width=width[i],
                                  height=height[i], linewidth=1, color='orchid', alpha=node_alpha[i], fill=True,
                                  label=graph.sentence_tokens[0][i])
                ax.add_patch(patch)

            nx.draw_networkx_labels(G, pos=pos, labels={i: graph.sentence_tokens[0][i] for i in range(graph.num_nodes)},
                                    font_size=self.vis_dict['font_size'], font_weight='bold', font_color='k')
            if not counter_edge_index is None:
                nx.draw_networkx_edges(G, pos=pos, edgelist=list(counter_edge_index.cpu().numpy().T),
                                       edge_color='mediumturquoise', width=self.vis_dict['width'] / 2.0, arrows=True,
                                       connectionstyle="arc3,rad=0.2")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        if 'Motif' in self.model_name:
            plt.figure(figsize=(8, 6), dpi=100)
            ax = plt.gca()
            pos = graph.pos[0]
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx}, nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'], node_color=graph.z[0][node_pos_idx], alpha=1,
                                   cmap='winter', linewidths=self.vis_dict['linewidths'], edgecolors='red',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0]))
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_neg_idx}, nodelist=node_neg_idx,
                                   node_size=self.vis_dict['node_size'], node_color=graph.z[0][node_neg_idx], alpha=0.2,
                                   cmap='winter', linewidths=self.vis_dict['linewidths'], edgecolors='whitesmoke',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0]))
            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index.cpu().numpy().T), edge_color='whitesmoke',
                                   width=self.vis_dict['width'], arrows=False)
            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   # np.ones(len(edge_imp[edge_pos_mask])),
                                   width=self.vis_dict['width'], edge_cmap=cm.get_cmap('bwr'), edge_vmin=-vmax,
                                   edge_vmax=vmax, arrows=False)
            if not counter_edge_index is None:
                nx.draw_networkx_edges(G, pos=pos, edgelist=list(counter_edge_index.cpu().numpy().T),
                                       edge_color='mediumturquoise', width=self.vis_dict['width'] / 3.0, arrows=False)

        if 'Mutag' in self.model_name:
            from rdkit.Chem.Draw import rdMolDraw2D
            idx = [int(i / 2) for i in idx]
            x = graph.x.detach().cpu().tolist()
            edge_index = graph.edge_index.T.detach().cpu().tolist()
            edge_attr = graph.edge_attr.detach().cpu().tolist()
            mol = graph_to_mol(x, edge_index, edge_attr)
            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            hit_at = np.unique(graph.edge_index[:, idx].detach().cpu().numpy()).tolist()

            hit_bonds = []
            for (u, v) in graph.edge_index.T[idx]:
                hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_at, highlightBonds=hit_bonds,
                                               highlightAtomColors={i: (0, 1, 0) for i in hit_at},
                                               highlightBondColors={i: (0, 1, 0) for i in hit_bonds})
            d.FinishDrawing()
            bindata = d.GetDrawingText()
            iobuf = io.BytesIO(bindata)
            image = Image.open(iobuf)
            image.show()
            if save:
                if name:
                    d.WriteDrawingText('image/%s/%s-%d-%s.png' % (self.model_name, name, int(graph.y[0]), self.name))
                else:
                    d.WriteDrawingText(
                        'image/%s/%s-%d-%s.png' % (self.model_name, str(graph.name[0]), int(graph.y[0]), self.name))
            return

        if 'MNIST' in self.model_name:
            plt.figure(figsize=(6, 6), dpi=100)
            ax = plt.gca()
            pos = graph.pos.detach().cpu().numpy()
            row, col = graph.edge_index
            z = np.zeros(graph.num_nodes)
            for i in idx:
                z[row[i]] += edge_imp[i]
                z[col[i]] += edge_imp[i]
            z = z / max(z)

            row, col = graph.edge_index
            pos = graph.pos.detach().cpu().numpy()
            z = graph.x.detach().cpu().numpy()
            edge_mask = torch.tensor(graph.x[row].view(-1) * graph.x[col].view(-1), dtype=torch.bool).view(-1)

            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index.cpu().numpy().T), edge_color='whitesmoke',
                                   width=self.vis_dict['width'], arrows=False)
            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index[:, edge_mask].cpu().numpy().T),
                                   edge_color='black', width=self.vis_dict['width'], arrows=False)
            nx.draw_networkx_nodes(G, pos=pos, node_size=self.vis_dict['node_size'], node_color='black', alpha=graph.x,
                                   linewidths=self.vis_dict['linewidths'], edgecolors='black')
            nx.draw_networkx_edges(G, pos=pos, edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]), width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('YlOrRd'), edge_vmin=-vmax, edge_vmax=vmax, arrows=False)
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx}, nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'], node_color='brown', alpha=z[node_pos_idx],
                                   linewidths=self.vis_dict['linewidths'], edgecolors='black')
            if not counter_edge_index is None:
                nx.draw_networkx_edges(G, pos=pos, edgelist=list(counter_edge_index.cpu().numpy().T),
                                       edge_color='mediumturquoise', width=self.vis_dict['width'] / 3.0, arrows=False)
        if self.model_name == "VGNet":
            from visual_genome import local as vgl
            idx = np.argsort(-edge_imp)[:topk]
            top_edges = graph.edge_index[:, idx]

            scene_graph = vgl.get_scene_graph(image_id=int(graph.name), images='visual_genome/raw',
                                              image_data_dir='visual_genome/raw/by-id/',
                                              synset_file='visual_genome/raw/synsets.json')
            # scene_graph = api.get_scene_graph_of_image(id=int(g.id))
            r = 0.95  # transparency
            img = Image.open("node_features/VG/raw/%d-%d.jpg" % (graph.name, graph.y))
            data = list(img.getdata())
            ndata = list(
                [(int((255 - p[0]) * r + p[0]), int((255 - p[1]) * r + p[1]), int((255 - p[2]) * r + p[2])) for p in
                 data])
            mode = img.mode
            width, height = img.size
            edges = list(top_edges.T)
            for i, (u, v) in enumerate(edges[::-1]):
                r = 1.0 - 1.0 / len(edges) * (i + 1)
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                for obj in [obj1, obj2]:
                    for x in range(obj.x, obj.width + obj.x):
                        for y in range(obj.y, obj.y + obj.height):
                            ndata[y * width + x] = (int((255 - data[y * width + x][0]) * r + data[y * width + x][0]),
                                                    int((255 - data[y * width + x][1]) * r + data[y * width + x][1]),
                                                    int((255 - data[y * width + x][2]) * r + data[y * width + x][2]))

            img = Image.new(mode, (width, height))
            img.putdata(ndata)

            plt.imshow(img)
            ax = plt.gca()
            for i, (u, v) in enumerate(edges):
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                ax.annotate("", xy=(obj2.x, obj2.y), xytext=(obj1.x, obj1.y),
                            arrowprops=dict(width=topk - i, color='wheat', headwidth=5))
                for obj in [obj1, obj2]:
                    ax.text(obj.x, obj.y - 8, str(obj), style='italic', fontsize=13,
                            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3,
                                  'edgecolor': rec_color[i % len(rec_color)]})
                    ax.add_patch(Rectangle((obj.x, obj.y), obj.width, obj.height, fill=False,
                                           edgecolor=rec_color[i % len(rec_color)], linewidth=1.5))
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.axis('off')
        if save:
            if name:
                plt.savefig(folder / Path(r'%s-%d-%s.png' % (name, int(graph.y[0]), self.name)), dpi=500,
                            bbox_inches='tight')
            else:
                if isinstance(graph.name[0], str):
                    plt.savefig(folder / Path(r'%s-%d-%s.png' % (str(graph.name[0]), int(graph.y[0]), self.name)),
                                dpi=500, bbox_inches='tight')
                else:
                    plt.savefig(folder / Path(r'%d-%d-%s.png' % (int(graph.name[0]), int(graph.y[0]), self.name)),
                                dpi=500, bbox_inches='tight')

        plt.show()
        

class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict(
            [('lin1', Lin(in_channels, hidden_channels)), ('act', act), ('lin2', Lin(hidden_channels, out_channels))]))

    def forward(self, x):
        return self.mlp(x)


class EdgeMaskNet(torch.nn.Module):

    def __init__(self, n_in_channels, e_in_channels, hid=72, n_layers=3):
        super(EdgeMaskNet, self).__init__()

        self.node_lin = Lin(n_in_channels, hid)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = ARMAConv(in_channels=hid, out_channels=hid)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid))

        if e_in_channels > 1:
            self.edge_lin1 = Lin(2 * hid, hid)
            self.edge_lin2 = Lin(e_in_channels, hid)
            self.mlp = MLP(2 * hid, hid, 1)
        else:
            self.mlp = MLP(2 * hid, hid, 1)
        self._initialize_weights()

    def forward(self, x, edge_index, edge_attr):
        x = torch.flatten(x, 1, -1)
        x = F.relu(self.node_lin(x))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index))
            x = batch_norm(x)

        e = torch.cat([x[edge_index[0, :]], x[edge_index[1, :]]], dim=1)

        if edge_attr.size(-1) > 1:
            e1 = self.edge_lin1(e)
            e2 = self.edge_lin2(edge_attr)
            e = torch.cat([e1, e2], dim=1)  # connection

        return self.mlp(e)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
EPS = 1e-6


class ReFine(Explainer):
    coeffs = {'edge_size': 1e-4, 'edge_ent': 1e-2, }

    def __init__(self, device, gnn_model, n_in_channels=14, e_in_channels=3, hid=50, n_layers=2, n_label=2, gamma=1):
        super(ReFine, self).__init__(device, gnn_model)
        # each class of label has an edge mask network
        self.edge_mask = nn.ModuleList(
            [EdgeMaskNet(n_in_channels, e_in_channels, hid=hid, n_layers=n_layers) for _ in range(n_label)]).to(device)
        self.gamma = gamma

    def __set_masks__(self, mask, model):
        # add mask to model
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = mask

    def __clear_masks__(self, model):
        # delete mask in model
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    def __reparameterize__(self, log_alpha, beta=1, training=True):
        # Equation 5 in the paper, calculate the prob of M_ij
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def fidelity_loss(self, log_logits, mask, pred_label):
        idx = [i for i in range(len(pred_label))]
        loss = -log_logits.softmax(dim=1)[idx, pred_label.view(-1)].sum()

        # make constrains on loss to avoid a too large sum of mask predictions
        loss = loss + self.coeffs['edge_size'] * mask.mean()

        # make constrains on loss to make the mask predictions close to 0 or 1
        ent = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def pack_subgraph(self, graph, imp, top_ratio=0.2):
        if abs(top_ratio - 1.0) < EPS:
            return graph, imp

        exp_subgraph = copy.deepcopy(graph)
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]

        # extract ego g
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(top_ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])

        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, _ = self.__relabel__(exp_subgraph,
                                                                                          exp_subgraph.edge_index)
        return exp_subgraph, imp[top_idx]

    def get_contrastive_loss(self, c, y, batch, tau=0.1):
        # normalize the graph representations
        c = c / c.norm(dim=1, keepdim=True)

        # compute the similarity score, which is between 0-1 with dim of (B,B)
        mat = F.relu(torch.mm(c, c.T))

        unique_graphs = torch.unique(batch)

        # obtain the final score matrix, dim is (B)
        ttl_scores = torch.sum(mat, dim=1)

        # InfoNCE loss https://lilianweng.github.io/posts/2021-05-31-contrastive/
        pos_scores = torch.tensor([mat[i, y == y[i]].sum() for i in unique_graphs]).to(c.device)
        contrastive_loss = - torch.logsumexp(pos_scores / (tau * ttl_scores), dim=0)
        return contrastive_loss

    def get_mask(self, graph):
        # obtain edges' batch index
        graph_map = graph.batch[graph.edge_index[0, :]]

        mask = torch.FloatTensor([]).to(graph.x.device)
        for i in range(len(graph.y)):
            edge_indicator = (graph_map == i).bool()

            # feed into the edge mask network to obtain the mask
            G_i_mask = self.edge_mask[graph.y[i]](graph.x, graph.edge_index[:, edge_indicator],
                                                  graph.edge_attr[edge_indicator, :]).view(-1)
            mask = torch.cat([mask, G_i_mask])
        # mask's dim is (B,N)
        return mask

    def get_pos_edge(self, graph, mask, ratio):
        """ obtain index of edges whose mask probabilities are the maximum """
        num_edge = [0]
        num_node = [0]
        sep_edge_idx = []
        # get batch index of each edge's nodes
        graph_map = graph.batch[graph.edge_index[0, :]]

        pos_idx = torch.LongTensor([])
        mask = mask.detach().cpu()
        # iterate through all graphs in batch
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()

            # the number of graph i's edges
            Gi_n_edge = len(edge_indicator)

            # compute the maximum number of edges under a given ratio, note the explainer is edge-view
            topk = max(math.ceil(ratio * Gi_n_edge), 1)

            # find the index of top-k edges with the largest mask prediction values
            Gi_pos_edge_idx = np.argsort(-mask[edge_indicator])[:topk]

            pos_idx = torch.cat([pos_idx, edge_indicator[Gi_pos_edge_idx]])
            # accumulate the number of edges and nodes within a batch
            num_edge.append(num_edge[i] + Gi_n_edge)
            num_node.append(num_node[i] + (graph.batch == i).sum().long())
            sep_edge_idx.append(Gi_pos_edge_idx)

        return pos_idx, num_edge, num_node, sep_edge_idx

    def explain_graph(self, graph, ratio=1.0, lr=1e-4, epoch=50, draw_graph=0, vis_ratio=0.2):
        edge_mask = self.get_mask(graph)
        edge_mask = self.__reparameterize__(edge_mask, training=False)
        imp = edge_mask.detach().cpu().numpy()
        self.last_result = (graph, imp)

        if draw_graph:
            self.visualize(graph, imp, vis_ratio=vis_ratio)
        return imp

    def pretrain(self, graph, ratio=1.0, reparameter=False, **kwargs):
        # compute saliency map M
        ori_mask = self.get_mask(graph)
        edge_mask = self.__reparameterize__(ori_mask, training=reparameter)

        # (1) compute fidelity loss
        self.__set_masks__(edge_mask, self.model)
        log_logits = self.model(graph)
        fid_loss = self.fidelity_loss(log_logits, edge_mask, graph.y)
        self.__clear_masks__(self.model)

        # (2) compute contrastive loss
        # obtain the index of edges that are kept
        pos_idx, _, _, _ = self.get_pos_edge(graph, edge_mask, ratio)
        pos_edge_mask = edge_mask[pos_idx]
        pos_edge_index = graph.edge_index[:, pos_idx]
        # pos_edge_attr = graph.edge_attr[pos_idx, :]
        self.__set_masks__(pos_edge_mask, self.model)

        # obtain sub-graphs' x/edge index/batch index/pos, compute the features of sub-graphs, dim is (B, D)，
        # G1_x, G1_pos_edge_index, G1_batch, G1_pos = self.__relabel__(graph, pos_edge_index)
        G1 = graph.clone()
        sub_nodes = torch.unique(pos_edge_index)
        G1.edge_index, G1.edge_attr = subgraph(sub_nodes, G1.edge_index, G1.edge_attr, num_nodes=G1.num_nodes, relabel_nodes=True)
        pos = None
        try:
            pos = G1.pos[sub_nodes]
        except:
            pass
        
        G1.pos = pos
        G1.x = G1.x[sub_nodes]
        # G1.adj_padded = G1.adj_padded[sub_nodes]
        # G1.x_padded = G1.x_padded[sub_nodes]
        G1.batch = G1.batch[sub_nodes]

        graph_rep = self.model.get_graph_rep(G1)
        # graph_rep = self.model.get_graph_rep(x=G1_x, edge_index=G1_pos_edge_index, edge_attr=pos_edge_attr,
        #                                      batch=G1_batch, pos=G1_pos)
        if isinstance(graph_rep, tuple): graph_rep = graph_rep[0]
        cts_loss = self.get_contrastive_loss(graph_rep, graph.y, graph.batch)
        self.__clear_masks__(self.model)
        del graph_rep
        del G1
        loss = fid_loss + self.gamma * cts_loss
        return loss

    def remap_device(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.edge_mask = self.edge_mask.to(device)
