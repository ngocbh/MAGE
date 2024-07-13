import os
import pickle
import json
import time
import torch
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import networkx as nx
# from dig.xgraph.method import GNNExplainer
from baselines.utils.gnn_explainer import GNNExplainer
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

    # bbbp warning
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
    else:
        node_indices_mask = (dataset.data.y != 0) * dataset.data.test_mask
        node_indices = torch.where(node_indices_mask)[0]

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

    gnn_explainer = GNNExplainer(model,
                                 epochs=config.explainers.param.epochs,
                                 lr=config.explainers.param.lr,
                                 coff_edge_size=config.explainers.param.coff_size,
                                 coff_edge_ent=config.explainers.param.coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer_perturb = GNNExplainer(model,
                                 epochs=config.explainers.param.epochs,
                                 lr=config.explainers.param.lr,
                                 coff_edge_size=config.explainers.param.coff_size,
                                 coff_edge_ent=config.explainers.param.coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer.device = device

    x_collector = XCollector()
    num_correct, num_examples  = 0, 0
    if config.models.param.graph_classification:
        for i, data in tqdm(enumerate(dataset[test_indices])):
            ori_data = data.clone()
            num_examples += 1
            data.to(device)
            prob = model(data).softmax(dim=-1)
            prediction = prob.argmax(-1).item()
            logger.info(f"explaining example {test_indices[i]}.")
            logger.info(f"prediction: {prediction} (prob: {prob[:, prediction].item():.3f}) | true label: {data.y.item()}")
            if prediction != data.y.item():
                continue
            num_correct += 1
            # Adding self loop during explaining while do not add self loop during training makes the model see out-of-distribution samples -> the explanations are not consistent, perforamance of the model decreases. We should keep the model and data as the way it is.
            # data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

            if config.datasets.ground_truth_available:
                _max_ex_size, _num_motifs = choose_explainer_param(data, dataset)
                max_ex_size = _max_ex_size if config.explainers.param.max_ex_size == -1 else config.explainers.param.max_ex_size
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                sparsity = _max_ex_size / data.num_nodes if config.explainers.param.sparsity == -1 else config.explainers.param.sparsity
            else:
                max_ex_size = config.explainers.param.max_ex_size
                num_motifs = config.experiments.num_motifs
                sparsity = config.explainers.param.sparsity

            if os.path.isfile(os.path.join(result_dir, f'example_{test_indices[i]}.pt')) and not config.rerun:
                edge_masks = torch.load(os.path.join(result_dir, f'example_{test_indices[i]}.pt'))
                edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
                print(f"load example {test_indices[i]}.")
                edge_masks, _, related_preds = \
                    gnn_explainer(data.x, data.edge_index,
                                  sparsity=sparsity,
                                  num_classes=dataset.num_classes,
                                  edge_masks=edge_masks)

            else:
                start_time = time.time()
                edge_masks, _, related_preds = \
                    gnn_explainer(data.x, data.edge_index,
                                  sparsity=sparsity,
                                  num_classes=dataset.num_classes)
                edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
                end_time = time.time()
                logger.info(f"explanation_time: {end_time - start_time:.3f}")
                related_preds[prediction]['running_time'] = end_time - start_time

                torch.save(edge_masks, os.path.join(result_dir, f'example_{test_indices[i]}.pt'))

            related_preds = related_preds[prediction]
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction

            edge_masks = [edge_mask.cpu().numpy() for edge_mask in edge_masks]
            expl_graph = None
            if config.datasets.ground_truth_available:
                gt_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
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
            else:
                gt_edge_mask = None
                
            if config.datasets.ground_truth_available:
                gt_coalition = [list(cpn) for cpn in nx.connected_components(other_info['gt_graph'])]
                logger.info(f"----> explanation precision: {stats['precision']} | recall: {stats['recall']} | f1_score: {stats['f1_score']} | ami: {stats['ami_score']}")
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

                graph = to_networkx(ori_data, to_undirected=True)
                plot_utils.plot(
                    graph,
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
    logger.info(f"Model accuracy: {num_correct/num_examples}")

    with open(os.path.join(result_dir, 'x_collector.pickle'), 'wb') as f:
        pickle.dump(x_collector, f)

    recorder = Recorder(config.record_filename)
    recorder.append(experiment_settings=experiment_settings,
                    experiment_data=experiment_data)
    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=gnn_explainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
