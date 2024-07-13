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
import optuna
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict

from utils import check_dir, fix_random_seed, Recorder, get_logger, XCollector
from utils.explainer_utils import explanation_filter, choose_explainer_param
from utils.evaluate_utils import compute_explanation_stats
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from utils.visualize import PlotUtils
from baselines.utils.gstarx import GStarX
from baselines.utils.gstarx import evaluate_coalition, scores2coalition, to_networkx
from mage.utils.gnn_helpers import get_reward_func_for_gnn_gc
from tqdm import tqdm

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

    # Test prediction accuracy and get average payoff (probability)
    preds = []
    rst = []
    for data in loader["test"]:
        data.to(device)
        pred = model(data).softmax(-1)
        preds += [pred]
        rst += [pred.argmax(-1) == data.y]
    preds = torch.concat(preds)
    rst = torch.concat(rst)
    payoff_avg = preds.mean(0).tolist()
    acc = rst.float().mean().item()
    logger.debug("Predicted prob: " + ",".join([f"{p:.4f}" for p in payoff_avg]))
    logger.debug(f"Test acc: {acc:.4f}")

    explainer = GStarX(
        model,
        device=device,
        max_sample_size=config.explainers.param.max_sample_size,
        tau=config.explainers.param.tau,
        payoff_type=config.explainers.param.payoff_type,
        payoff_avg=payoff_avg,
        subgraph_building_method=config.datasets.subgraph_building_method,
    )

    if config.models.param.graph_classification:
        x_collector = XCollector()
        for i, data in enumerate(dataset[test_indices]):
            # if test_indices[i] != 4873:
            #     continue
            data.to(device)
            ori_data = data.clone()
            prediction = model(data).argmax(-1).item()
            if prediction != data.y.item():
                continue
            logger.info(f"explaining example {test_indices[i]}.")
            logger.info(f"prediction: {prediction} | true label: {data.y.item()}")
            logger.info(f"num_nodes: {data.num_nodes}")
            # data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

            if config.datasets.ground_truth_available:
                _max_ex_size, _num_motifs = choose_explainer_param(data, dataset)
                max_ex_size = _max_ex_size if config.explainers.param.max_ex_size == -1 else config.explainers.param.max_ex_size
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
            else:
                max_ex_size = config.explainers.param.max_ex_size
                num_motifs = 1 if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                if max_ex_size < 1:
                    # if the number of nodes is too small, still allow highlighting 3 nodes
                    max_ex_size = max(3, int(max_ex_size * data.num_nodes))

            logger.info(f"running with - max_ex_size: {max_ex_size} | num_motifs: {num_motifs}")
            explained_example_path = os.path.join(result_dir, f'example_{test_indices[i]}.pt')

            if os.path.isfile(os.path.join(result_dir, f'example_{test_indices[i]}.pt')) and not config.rerun:
                node_scores, related_preds = torch.load(explained_example_path)
                logger.info(f"load example {test_indices[i]}.")
            else:
                # use data.num_nodes as num_samples
                related_preds = {}
                start_time = time.time()
                node_scores = explainer.explain(
                    data,
                    superadditive_ext=config.explainers.superadditive_ext,
                    sample_method=config.explainers.sample_method,
                    num_samples=config.explainers.num_samples,
                    k=config.explainers.num_hops,
                )
                end_time = time.time()
                related_preds['running_time'] = end_time - start_time
                torch.save((node_scores, related_preds), explained_example_path)

            coalition = scores2coalition(node_scores, max_ex_size)
            f, inv_f, sp = evaluate_coalition(explainer, data, coalition,
                                              config.datasets.subgraph_building_method)
            related_preds['fidelity'] = f
            related_preds['fidelity_inv'] = inv_f
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction
            logger.info(f"coalition: {coalition}")
            logger.info(f"fid: {f} | inv_fid: {inv_f}")

            edge_index = ori_data.edge_index.clone()
            edge_mask = edge_index[0].cpu().apply_(lambda x: x in coalition).bool() & \
                        edge_index[1].cpu().apply_(lambda x: x in coalition).bool()
            edge_mask = edge_mask.float().numpy()

            if config.datasets.ground_truth_available:
                gt_edge_mask = dataset.gen_motif_edge_mask(ori_data).float().cpu().numpy()
            else:
                gt_edge_mask = None

            predict_fn = get_reward_func_for_gnn_gc(model, prediction, payoff_type='prob')
            stats, other_info = compute_explanation_stats(ori_data, gt_edge_mask,
                                                          edge_mask=edge_mask, node_list=coalition,
                                                          num_motifs=num_motifs, max_nodes=max_ex_size,
                                                          predict_fn=predict_fn,
                                                          subgraph_building_method=config.datasets.subgraph_building_method)
            for key, value in stats.items():
                related_preds[key] = value


            if config.datasets.ground_truth_available:
                logger.info(f"----> explanation precision: {stats['precision']} | recall: {stats['recall']} | f1_score: {stats['f1_score']}")
                logger.info(f"----> related_preds: {related_preds}")
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
                    nodelist=[coalition],
                    x=data.x,
                    words=words,
                    title_sentence=title_sentence,
                    figname=vis_name,
                    data=data,
                )

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

