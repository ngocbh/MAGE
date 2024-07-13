import os
import pickle
import json
import time
import torch
import hydra
from torch import Tensor
from typing import List, Dict, Tuple
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import numpy as np
from dig.xgraph.utils.compatibility import compatible_state_dict
from torch_geometric.utils import add_remaining_self_loops

from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader, SynGraphDataset
from utils import check_dir, fix_random_seed, Recorder, get_logger, XCollector
from utils.explainer_utils import explanation_filter, choose_explainer_param
from utils.evaluate_utils import compute_explanation_stats, get_explanation_syn
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from utils.visualize import PlotUtils
from mage.utils.gnn_helpers import to_networkx, get_reward_func_for_gnn_gc

from dig.xgraph.method import PGExplainer
from dig.xgraph.method.base_explainer import ExplainerBase

class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        pred_label = kwargs.get('pred_label')
        num_classes = kwargs.get('num_classes')
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        # edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False)
            # edge_masks
            edge_masks = [edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid()
                               for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(x, edge_index)
            with torch.no_grad():
                if self.explain_graph:
                    related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)

            self.__clear_masks__()

        else:
            node_idx = kwargs.get('node_idx')
            sparsity = kwargs.get('sparsity')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            select_edge_index = torch.arange(0, edge_index.shape[1])
            subgraph_x, subgraph_edge_index, _, subset, kwargs = \
                self.explainer.get_subgraph(node_idx, x, edge_index, select_edge_index=select_edge_index)
            select_edge_index = kwargs['select_edge_index']
            self.select_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                         device=self.device,
                                                         dtype=torch.bool)
            self.select_edge_mask.fill_(False)
            self.select_edge_mask[select_edge_index] = True
            self.hard_edge_mask = edge_index.new_empty(subgraph_edge_index.size(1),
                                                       device=self.device,
                                                       dtype=torch.bool)
            self.hard_edge_mask.fill_(True)
            self.subset = subset
            self.new_node_idx = torch.where(subset == node_idx)[0]

            subgraph_embed = self.model.get_emb(subgraph_x, subgraph_edge_index)
            _, subgraph_edge_mask = self.explainer.explain(subgraph_x,
                                                           subgraph_edge_index,
                                                           embed=subgraph_embed,
                                                           tmp=1.0,
                                                           training=False,
                                                           node_idx=self.new_node_idx)

            # edge_masks
            edge_masks = [subgraph_edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [
                self.control_sparsity(subgraph_edge_mask, sparsity=sparsity).sigmoid()
                for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(subgraph_x, subgraph_edge_index)
            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    subgraph_x, subgraph_edge_index, hard_edge_masks, node_idx=self.new_node_idx)

            self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds

def pipeline(config):
    explainer_name = (
        f"{config.explainers.explainer_name}"
    )

    experiment_settings = [explainer_name] + config.experiments.experiment_settings
    experiment_name = "_".join(experiment_settings)

    result_dir = os.path.join(config.result_dir,
                              experiment_name)

    log_file = (
        f"{experiment_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    log_mode = 'w' if config.rerun else 'a'
    logger = get_logger(result_dir, log_file, config.console_log, config.log_level, log_mode)
    logger.info(f"Conducting experiment: {experiment_name}")
    logger.info(OmegaConf.to_yaml(config))
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))

    logger.info(f"saving results at {result_dir}")

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        train_indices = loader['train'].dataset.indices
        test_indices = loader['test'].dataset.indices
        test_indices = explanation_filter(dataset.name, dataset, test_indices)
    else:
        train_indices = range(len(dataset))

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)
    eval_model = get_gnnNets(input_dim=dataset.num_node_features,
                             output_dim=dataset.num_classes,
                             model_config=config.models)

    state_dict = torch.load(os.path.join(config.models.gnn_saving_dir,
                                         config.datasets.dataset_name,
                                         f"{config.models.gnn_name}_"
                                         f"{len(config.models.param.gnn_latent_dim)}l_best.pth"))['net']

    state_dict = compatible_state_dict(state_dict)
    model.load_state_dict(state_dict)
    eval_model.load_state_dict(state_dict)

    model.to(device)
    eval_model.to(device)
    model.eval()
    eval_model.eval()

    check_dir(result_dir)
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)

    if config.models.param.graph_classification:
        if config.models.param.concate:
            input_dim = sum(config.models.param.gnn_latent_dim) * 2
        else:
            input_dim = config.models.param.gnn_latent_dim[-1] * 2
    else:
        if config.models.param.concate:
            input_dim = sum(config.models.param.gnn_latent_dim) * 3
        else:
            input_dim = config.models.param.gnn_latent_dim[-1] * 3

    pgexplainer = PGExplainer(model,
                              in_channels=input_dim,
                              device=device,
                              explain_graph=config.models.param.graph_classification,
                              epochs=config.explainers.param.ex_epochs,
                              lr=config.explainers.param.ex_learning_rate,
                              coff_size=config.explainers.param.coff_size,
                              coff_ent=config.explainers.param.coff_ent,
                              # sample_bias=config.explainers.param.sample_bias,
                              t0=config.explainers.param.t0,
                              t1=config.explainers.param.t1)

    pgexplainer_saving_path = os.path.join(result_dir,
                                           config.explainers.explainer_saving_name)

    if os.path.isfile(pgexplainer_saving_path) and not config.rerun:
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        state_dict = compatible_state_dict(state_dict)
        pgexplainer.load_state_dict(state_dict)
    else:
        if config.models.param.graph_classification:
            pgexplainer.train_explanation_network(dataset[train_indices])
        else:
            pgexplainer.train_explanation_network(dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        state_dict = compatible_state_dict(state_dict)
        pgexplainer.load_state_dict(state_dict)

    index = 0
    x_collector = XCollector()
    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer,
                                          model=eval_model,
                                          molecule=True)
    pgexplainer_edges.device = pgexplainer.device

    num_correct, num_examples  = 0, 0
    if config.models.param.graph_classification:
        for i, data in enumerate(dataset[test_indices]):
            num_examples += 1
            ori_data = data.clone()
            data.to(device)
            prob = model(data).softmax(dim=-1)
            prediction = prob.argmax(-1).item()
            logger.info(f"explaining example {test_indices[i]}.")
            logger.info(f"prediction: {prediction} (prob: {prob[:, prediction].item():.3f}) | true label: {data.y.item()}")
            if prediction != data.y.item():
                continue
            num_correct += 1
            # data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

            if config.datasets.ground_truth_available:
                _max_ex_size, _num_motifs = choose_explainer_param(data, dataset)
                max_ex_size = _max_ex_size if config.explainers.param.max_ex_size == -1 else config.explainers.param.max_ex_size
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                sparsity = 1 - _max_ex_size / data.num_nodes if config.explainers.param.sparsity == -1 else config.explainers.param.sparsity
            else:
                max_ex_size = config.explainers.param.max_ex_size
                num_motifs = 1 if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                if max_ex_size < 1:
                    max_ex_size = max(3, int(max_ex_size * data.num_nodes))
                sparsity = 1 - max_ex_size / data.num_nodes

            start_time = time.time()
            edge_masks, hard_edge_masks, related_preds = \
                pgexplainer_edges(data.x, data.edge_index,
                                  num_classes=dataset.num_classes,
                                  sparsity=sparsity)
            end_time = time.time()

            related_preds[prediction]['running_time'] = end_time - start_time
            edge_masks = [edge_mask.detach().cpu().numpy() for edge_mask in edge_masks]

            related_preds = related_preds[prediction]
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction

            expl_graph = None
            if config.datasets.ground_truth_available:
                gt_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
            else:
                gt_edge_mask = None

            predict_fn = get_reward_func_for_gnn_gc(model, prediction, payoff_type='prob')
            stats, other_info = compute_explanation_stats(data, gt_edge_mask,
                                                          edge_mask=edge_masks[prediction], node_list=None,
                                                          num_motifs=num_motifs, max_nodes=max_ex_size,
                                                          predict_fn=predict_fn,
                                                          subgraph_building_method=config.datasets.subgraph_building_method)
            for key, value in stats.items():
                related_preds[key] = value
            expl_graph = other_info['expl_graph']
            logger.info(f"{stats}")

            if config.datasets.ground_truth_available:
                title_sentence = f'prec: {related_preds["precision"]:.3f}, ' \
                                 f'rec: {(related_preds["recall"]):.3f}, ' \
                                 f'f1: {related_preds["f1_score"]:.3f}'
            else:
                title_sentence = None 

            if hasattr(dataset, 'supplement'):
                words = dataset.supplement['sentence_tokens'][str(test_indices[i])]
            else:
                words = None

            if config.visualize:
                predict_true = 'True' if prediction == data.y.item() else "False"
                vis_name = os.path.join(result_dir,
                                        f'example_{test_indices[i]}_'
                                        f'prediction_{prediction}_'
                                        f'label_{data.y.item()}_'
                                        f'pred_{predict_true}.png')

                graph = to_networkx(ori_data, to_undirected=True)
                plot_utils.plot(
                    graph,
                    expl_graph=expl_graph,
                    x=data.x,
                    title_sentence=title_sentence,
                    words=words,
                    figname=vis_name,
                    data=data,
                )

            x_collector.collect_data(related_preds)

            if config.max_ins != -1 and i >= config.max_ins - 1:
                break

    experiment_data = x_collector.get_summarized_results()
    logger.info(json.dumps(experiment_data, indent=4))
    logger.info(f"Model accuracy: {num_correct/num_examples}")

    with open(os.path.join(result_dir, 'x_collector.pickle'), 'wb') as f:
        pickle.dump(x_collector, f)

    recorder = Recorder(config.record_filename)
    recorder.append(experiment_settings=experiment_settings,
                    experiment_data=experiment_data)
    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=pgexplainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explainer_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
