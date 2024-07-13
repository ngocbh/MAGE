import os
import json
import time
import torch
import pickle
import networkx as nx
from omegaconf import OmegaConf
from dig.xgraph.utils.compatibility import compatible_state_dict

from mage import Mage
from mage.maskers import PyGDataMasker
from utils.visualize import PlotUtils
from mage.utils.gnn_helpers import to_networkx

from utils import check_dir, Recorder, get_logger
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from utils import XCollector
from utils.explainer_utils import explanation_filter, choose_explainer_param
from utils.evaluate_utils import compute_explanation_stats
from mage.utils.gnn_helpers import get_reward_func_for_gnn_gc


def pipeline(config):
    explainer_name = (
        f"{config.explainers.explainer_name}_{config.explainers.score_fn}"
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

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
    logger.info(f"Running on device: {device}")

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
        masker = PyGDataMasker(method=config.datasets.subgraph_building_method)
        explainer = Mage(model, masker=masker,
                         payoff_type=config.explainers.param.payoff_type,
                         device=device,
                         random_state=42)

        x_collector = XCollector()

        cnt = 0
        for i, data in enumerate(dataset[test_indices]):
            # if test_indices[i] != 6744:
            #     continue
            data.to(device)
            ori_data = data.clone()
            prob = model(data).softmax(dim=-1)
            prediction = prob.argmax(-1).item()
            if prediction != data.y.item() and config.datasets.ground_truth_available:
                continue
            cnt += 1
            # if cnt-1 != 182:
            #     continue

            logger.info(f"explaining example {test_indices[i]}:{cnt} | num_nodes: {data.num_nodes}.")
            logger.info(f"prediction: {prediction} (prob: {prob[:, prediction].item():.3f}) | true label: {data.y.item()}")

            # data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

            if config.datasets.ground_truth_available:
                _beta, _num_motifs = choose_explainer_param(data, dataset)
                beta = _beta if config.explainers.param.beta == -1 else config.explainers.param.beta
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                assert num_motifs >= 1
            else:
                beta = config.explainers.param.beta
                num_motifs = 1 if config.explainers.param.num_motifs == -1 else config.explainers.param.num_motifs
                if beta < 1:
                    beta = max(3, int(beta * data.num_nodes))

            logger.info(f"running with - beta: {beta} | num_motifs: {num_motifs}")

            if os.path.isfile(os.path.join(result_dir, f'example_{test_indices[i]}.pt')) and not config.rerun:
                motifs, info = torch.load(os.path.join(result_dir, f'example_{test_indices[i]}.pt'))
                logger.info(f"load example {test_indices[i]}.")
            else:
                start_time = time.time()
                motifs, info = explainer.explain(
                    input=data,
                    num_motifs=num_motifs,
                    target_class=prediction,
                    beta=beta,
                    tau=config.explainers.param.tau,
                    omega=config.explainers.param.omega,
                    ord=config.explainers.order,
                    method=config.explainers.score_fn,
                    num_samples=config.explainers.param.num_samples,
                    connectivity=config.explainers.param.connectivity,
                )
                explanation_time = time.time() - start_time
                logger.info(f"explanation_time: {explanation_time:.3f}")
            
            torch.save((motifs, info), os.path.join(result_dir, f'example_{test_indices[i]}.pt'))
            logger.info(f"motifs: {motifs}")

            related_preds = info['related_preds']
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction

            coalition = [u for c in motifs for u in c]
            edge_index = ori_data.edge_index.clone()
            edge_mask = edge_index[0].cpu().apply_(lambda x: x in coalition).bool() & \
                        edge_index[1].cpu().apply_(lambda x: x in coalition).bool()
            edge_mask = edge_mask.float().numpy()

            gt_coalition = None
            if config.datasets.ground_truth_available:
                gt_edge_mask = dataset.gen_motif_edge_mask(ori_data).float().cpu().numpy()
            else:
                gt_edge_mask = None

            predict_fn = get_reward_func_for_gnn_gc(model, prediction, payoff_type='prob')

            stats, other_info = compute_explanation_stats(ori_data, gt_edge_mask,
                                                          edge_mask=edge_mask, node_list=coalition,
                                                          num_motifs=num_motifs, max_nodes=beta,
                                                          predict_fn=predict_fn,
                                                          subgraph_building_method=config.datasets.subgraph_building_method)
            for key, value in stats.items():
                related_preds[key] = value

            if config.datasets.ground_truth_available:
                gt_coalition = [list(cpn) for cpn in nx.connected_components(other_info['gt_graph'])]
                logger.info(f"----> explanation precision: {stats['precision']} | recall: {stats['recall']} | f1_score: {stats['f1_score']} | ami: {stats['ami_score']}")
                title_sentence = f'prec: {related_preds["precision"]:.3f}, ' \
                                 f'rec: {(related_preds["recall"]):.3f}, ' \
                                 f'f1: {related_preds["f1_score"]:.3f}'
            else:
                title_sentence = None
                # title_sentence = f'\nfid_delta: {related_preds["fid_delta"]:.3f} ' \
                #                  f'fid_plus: {related_preds["fid_plus"]:.3f} ' \
                #                  f'fid_minus: {related_preds["fid_minus"]:.3f} ' \


            logger.info(f"----> related_preds: {related_preds}")

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
                    nodelist=motifs,
                    indices=info['normalized_indices'],
                    x=data.x,
                    words=words,
                    title_sentence=title_sentence,
                    figname=vis_name,
                    data=data,
                )
                if gt_coalition is not None:
                    vis_name = os.path.join(result_dir,
                                            f'example_{test_indices[i]}_'
                                            f'label_{data.y.item()}_'
                                            f'gt.png')

                    plot_utils.plot(
                        graph,
                        nodelist=gt_coalition,
                        data=data,
                        x=data.x,
                        words=words,
                        title_sentence=title_sentence,
                        figname=vis_name,
                    )

            x_collector.collect_data(related_preds)

            if config.max_ins != -1 and cnt >= config.max_ins - 1:
                break

    experiment_data = x_collector.get_summarized_results()
    logger.info(json.dumps(experiment_data, indent=4))

    with open(os.path.join(result_dir, 'x_collector.pickle'), 'wb') as f:
        pickle.dump(x_collector, f)

    recorder = Recorder(config.record_filename)
    recorder.append(experiment_settings=experiment_settings,
                    experiment_data=experiment_data)
    recorder.save()
