import os
import time
import torch
import json
import pickle
import numpy as np
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

from dig.xgraph.method import SubgraphX
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict

from utils import check_dir, fix_random_seed, Recorder, get_logger, XCollector
from utils.explainer_utils import explanation_filter, choose_explainer_param
from utils.evaluate_utils import compute_explanation_stats
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from utils.visualize import PlotUtils
from mage.utils.gnn_helpers import get_reward_func_for_gnn_gc


def pipeline(config):
    explainer_name = (
        f"{config.explainers.explainer_name}_{config.explainers.param.reward_method}"
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
        test_indices = loader['test'].dataset.indices
        test_indices = explanation_filter(dataset.name, dataset, test_indices)

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)

    state_dict = compatible_state_dict(torch.load(os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_"
        f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
    ))['net'])

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    check_dir(result_dir)
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)

    if config.models.param.graph_classification:
        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.datasets.subgraph_building_method,
                              save_dir=result_dir)
        x_collector = XCollector()

        # test_indices[0] = 641
        for i, data in enumerate(dataset[test_indices]):
            # if test_indices[i] != 4873:
            #     continue
            data.to(device)
            ori_data = data.clone()
            saved_MCTSInfo_list = None
            prediction = model(data).argmax(-1).item()
            if prediction != data.y.item():
                continue
            logger.info(f"explaining example {test_indices[i]}.")
            logger.info(f"prediction: {prediction} | true label: {data.y.item()}")
            # data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

            if config.datasets.ground_truth_available:
                _max_ex_size, _num_motifs = choose_explainer_param(data, dataset)
                max_ex_size = _max_ex_size if config.explainers.param.max_ex_size == -1 else config.explainers.param.max_ex_size
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
            else:
                max_ex_size = config.explainers.param.max_ex_size
                num_motifs = 1 if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                if max_ex_size < 1:
                    max_ex_size = max(3, int(max_ex_size * data.num_nodes))

            logger.info(f"running with - max_ex_size: {max_ex_size} | num_motifs: {num_motifs}")

            # if os.path.isfile(os.path.join(result_dir, f'example_{test_indices[i]}.pt')) and not config.rerun:
            try:
                saved_data = torch.load(os.path.join(result_dir, f'example_{test_indices[i]}.pt'))
                if isinstance(saved_data, tuple):
                    explain_result, related_preds = saved_data
                else:
                    explain_result, related_preds = \
                        subgraphx.explain(data.x, data.edge_index,
                                          max_nodes=max_ex_size,
                                          label=prediction,
                                          saved_MCTSInfo_list=saved_data)
                logger.info(f"load example {test_indices[i]}.")
            # else:
            except:
                start_time = time.time()
                explain_result, related_preds = \
                    subgraphx.explain(data.x, data.edge_index,
                                      max_nodes=max_ex_size,
                                      label=prediction,
                                      saved_MCTSInfo_list=None)
                end_time = time.time()
                logger.info(f"explanation_time: {end_time - start_time:.3f}")
                related_preds['running_time'] = end_time - start_time

                torch.save((explain_result, related_preds), os.path.join(result_dir, f'example_{test_indices[i]}.pt'))

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)

            explanation = find_closest_node_result(explain_result, max_nodes=max_ex_size)
            coalition = explanation.coalition
            logger.info(f"coalition explanation: {coalition}")
            edge_index = ori_data.edge_index.clone()
            edge_mask = edge_index[0].cpu().apply_(lambda x: x in coalition).bool() & \
                        edge_index[1].cpu().apply_(lambda x: x in coalition).bool()
            edge_mask = edge_mask.float().numpy()

            if config.datasets.ground_truth_available:
                gt_edge_mask = dataset.gen_motif_edge_mask(ori_data).float().cpu().numpy()
            else:
                gt_edge_mask = None

            predict_fn = get_reward_func_for_gnn_gc(model, prediction, payoff_type='prob')

            stats, _ = compute_explanation_stats(ori_data, gt_edge_mask,
                                                 edge_mask=edge_mask, node_list=coalition,
                                                 num_motifs=num_motifs, max_nodes=max_ex_size,
                                                 predict_fn=predict_fn,
                                                 subgraph_building_method=config.datasets.subgraph_building_method)
            for key, value in stats.items():
                related_preds[key] = value

            logger.info(f"coalition: {coalition}")
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction

            if config.datasets.ground_truth_available:
                logger.info(f"----> explanation precision: {stats['precision']} | recall: {stats['recall']} | f1_score: {stats['f1_score']}")
                title_sentence = f'prec: {related_preds["precision"]:.3f}, ' \
                                 f'rec: {(related_preds["recall"]):.3f}, ' \
                                 f'f1: {related_preds["f1_score"]:.3f}'
            else:
                title_sentence = None


            logger.info(f"----> related_preds: {related_preds}")

            if hasattr(dataset, 'supplement'):
                words = dataset.supplement['sentence_tokens'][str(test_indices[i])]
            else:
                words = None

            if config.visualize:
                predict_true = 'True' if prediction == data.y.item() else "False"
                tree_node_x = find_closest_node_result(explain_result, max_nodes=max_ex_size)
                predict_true = 'True' if prediction == data.y.item() else "False"
                vis_name = os.path.join(result_dir,
                                        f'example_{test_indices[i]}_'
                                        f'prediction_{prediction}_'
                                        f'label_{data.y.item()}_'
                                        f'pred_{predict_true}.png')

                plot_utils.plot(
                    tree_node_x.ori_graph,
                    nodelist=[tree_node_x.coalition],
                    x=data.x,
                    words=words,
                    title_sentence=title_sentence,
                    figname=vis_name,
                    data=data,
                )

            x_collector.collect_data(related_preds)

            if config.max_ins != -1 and i >= config.max_ins - 1:
                break
    else:
        x_collector = XCollector()
        data = dataset.data
        ori_data = data.clone()
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        data.to(device)
        predictions = model(data).argmax(-1)

        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.datasets.subgraph_building_method,
                              save_dir=result_dir)

        for i, node_idx in enumerate(node_indices):
            saved_MCTSInfo_list = None

            if os.path.isfile(os.path.join(result_dir, f'example_{node_idx}.pt')) and not config.rerun:
                saved_MCTSInfo_list = torch.load(os.path.join(result_dir,
                                                              f'example_{node_idx}.pt'))
                logger.info(f"load example {node_idx}.")

            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.param.max_ex_size,
                                  label=predictions[node_idx].item(),
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)

            torch.save(explain_result, os.path.join(result_dir, f'example_{node_idx}.pt'))

            title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
                             f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
                             f'spar: {related_preds["sparsity"]:.3f}'

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            # if isinstance(dataset, SynGraphDataset):
            #     explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.param.max_ex_size)
            #     edge_mask = edge_mask.float().numpy()
            #     motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
            #     accuracy = accuracy_score(edge_mask, motif_edge_mask)
            #     roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
            #     related_preds['accuracy'] = roc_auc

            if isinstance(dataset, SynGraphDataset):
                explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.param.max_ex_size)
                subset = subgraphx.mcts_state_map.subset
                coalition = subset[explanation.coalition]
                edge_index = ori_data.edge_index.clone().detach()
                edge_mask = edge_index[0].cpu().apply_(lambda x: x in coalition).bool() & \
                            edge_index[1].cpu().apply_(lambda x: x in coalition).bool()
                edge_mask = edge_mask.float().numpy()
                motif_edge_mask = dataset.gen_motif_edge_mask(ori_data, node_idx=node_idx)

                accuracy = accuracy_score(edge_mask, motif_edge_mask)
                # if np.count_nonzero(edge_mask) == 0:
                #     roc_auc = 0.
                # else:
                #     roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
                precision, recall, f1_score, _ = precision_recall_fscore_support(edge_mask, motif_edge_mask, average='binary')
                # related_preds['roc_auc'] = roc_auc
                related_preds['accuracy'] = accuracy 
                related_preds['precision'] = precision
                related_preds['recall'] = recall
                related_preds['f1_score'] = f1_score
                logger.info(f"----> explanation accuracy: {accuracy}")
                logger.info(f"----> explanation precision: {precision} | recall: {recall} | f1_score: {f1_score}")

            subgraphx.visualization(explain_result,
                                    y=data.y,
                                    max_nodes=config.explainers.param.max_ex_size,
                                    plot_utils=plot_utils,
                                    title_sentence=title_sentence,
                                    vis_name=os.path.join(result_dir,
                                                          f'example_{node_idx}.png'))
            x_collector.collect_data(related_preds)

            if config.max_ins != -1 and i >= config.max_ins - 1:
                break

    experiment_data = x_collector.get_summarized_results()
    logger.info(json.dumps(experiment_data, indent=4))

    with open(os.path.join(result_dir, 'x_collector.pickle'), 'wb') as f:
        pickle.dump(x_collector, f)

    recorder = Recorder(config.record_filename)
    recorder.append(experiment_settings=experiment_settings,
                    experiment_data=experiment_data)
    recorder.save()
