import os
import pickle
import json
import torch
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import add_remaining_self_loops

from dig.xgraph.utils.compatibility import compatible_state_dict

from utils import check_dir, fix_random_seed, Recorder, get_logger, XCollector
from utils.explainer_utils import explanation_filter, choose_explainer_param
from utils.evaluate_utils import compute_explanation_stats
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from utils.visualize import PlotUtils
from baselines.utils.grad_cam import GradCAM
from mage.utils.gnn_helpers import to_networkx, get_reward_func_for_gnn_gc


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

    gc_explainer = GradCAM(model, explain_graph=config.models.param.graph_classification)

    x_collector = XCollector()
    if config.models.param.graph_classification:
        for i, data in enumerate(dataset[test_indices]):
            # if test_indices[i] != 4873:
            #     continue
            data.to(device)
            prediction = model(data).argmax(-1).item()
            if prediction != data.y.item():
                continue
            logger.info(f"explaining example {test_indices[i]}.")
            logger.info(f"prediction: {prediction} | true label: {data.y.item()}")

            if config.datasets.ground_truth_available:
                _max_ex_size, _num_motifs = choose_explainer_param(data, dataset)
                max_ex_size = _max_ex_size if config.explainers.param.max_ex_size == -1 else config.explainers.param.max_ex_size
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                sparsity = 1 - max_ex_size / data.num_nodes
            else:
                max_ex_size = config.explainers.param.max_ex_size
                num_motifs = 1 if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                if max_ex_size < 1:
                    max_ex_size = max(3, int(max_ex_size * data.num_nodes))
                sparsity = 1 - max_ex_size / data.num_nodes

            if os.path.isfile(os.path.join(result_dir, f'example_{test_indices[i]}.pt')) and not config.rerun:
                edge_masks = torch.load(os.path.join(result_dir, f'example_{test_indices[i]}.pt'))
                edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
                print(f"load example {test_indices[i]}.")
                edge_masks, hard_edge_masks, related_preds = \
                    gc_explainer(data.x, data.edge_index,
                                 sparsity=sparsity,
                                 num_classes=dataset.num_classes,
                                 edge_masks=edge_masks)
            else:
                edge_masks, hard_edge_masks, related_preds = \
                    gc_explainer(data.x, data.edge_index,
                                 sparsity=sparsity,
                                 num_classes=dataset.num_classes)

                edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
                torch.save(edge_masks, os.path.join(result_dir, f'example_{test_indices[i]}.pt'))

            related_preds = related_preds[prediction]
            edge_masks = [edge_mask.cpu().numpy() for edge_mask in edge_masks]
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

            logger.info(f"explanation: {expl_graph.nodes}")
            logger.info(f"{stats}")
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction

            if config.datasets.ground_truth_available:
                title_sentence = f'prec: {related_preds["precision"]:.3f}, ' \
                                 f'rec: {(related_preds["recall"]):.3f}, ' \
                                 f'f1: {related_preds["f1_score"]:.3f}'
            else:
                title_sentence = None

            if config.visualize:
                predict_true = 'True' if prediction == data.y.item() else "False"
                vis_name = os.path.join(result_dir,
                                        f'example_{test_indices[i]}_'
                                        f'prediction_{prediction}_'
                                        f'label_{data.y.item()}_'
                                        f'pred_{predict_true}.png')

                graph = to_networkx(data, to_undirected=True)
                plot_utils.plot(
                    graph,
                    nodelist=[list(expl_graph.nodes)],
                    expl_graph=expl_graph,
                    x=data.x,
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


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=grad_cam')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
