import json
import os
import pickle
import time

import hydra
import numpy as np
import optuna
import torch
from dig.xgraph.utils.compatibility import compatible_state_dict
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_remaining_self_loops

from baselines.utils.grad_cam import GradCAM
from baselines.utils.refine import ReFine
from mage.utils.gnn_helpers import to_networkx, get_reward_func_for_gnn_gc
from dataset import get_dataloader, get_dataset
from gnnNets import get_gnnNets
from utils import Recorder, XCollector, check_dir, fix_random_seed, get_logger
from utils.evaluate_utils import compute_explanation_stats
from utils.explainer_utils import choose_explainer_param, explanation_filter
from utils.visualize import PlotUtils
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
    logger.debug(OmegaConf.to_yaml(config))
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
        val_indices = loader['eval'].dataset.indices
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

    explainer = ReFine(device, model, n_in_channels=dataset.num_features, e_in_channels=dataset.num_edge_features, n_label=dataset.num_classes, gamma=config.explainers.param.gamma)
    # gc_explainer = GradCAM(model, explain_graph=config.models.param.graph_classification)
    parameters = list()
    for k, edge_mask in enumerate(explainer.edge_mask):
        edge_mask.train()
        parameters += list(explainer.edge_mask[k].parameters())

    optimizer = torch.optim.Adam(parameters, lr=config.explainers.param.lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                    mode='min',
                                    factor=0.2,
                                    patience=3,
                                    min_lr=1e-5
                                    )

    loss_all = 0
    train_loader = DataLoader(dataset[train_indices], batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset[val_indices], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[test_indices], batch_size=1, shuffle=False)
    for epoch in range(config.explainers.param.epoch):
        
        for data in tqdm(train_loader):
            data = data.to(device)
            if data.edge_attr is None:
                data.edge_attr = torch.ones(data.edge_index.shape[1], 1).to(device)

            optimizer.zero_grad()
            loss = explainer.pretrain(
                data,
                ratio=config.explainers.param.sparsity,
                reperameter=True,
                temp=config.explainers.param.tau
                )
            loss.backward()
            optimizer.step()
            loss_all += loss.detach().cpu().item()
            del loss
            del data

        val_loss_all = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                if data.edge_attr is None:
                    data.edge_attr = torch.ones(data.edge_index.shape[1], 1).to(device)

                loss = explainer.pretrain(
                        data,
                        ratio=config.explainers.param.sparsity,
                        reperameter=False,
                        temp=config.explainers.param.tau
                        )
                val_loss_all += loss.detach().cpu().item()
                del loss
                del data

            scheduler.step(val_loss_all)
            lr = optimizer.param_groups[0]['lr']

        train_loss = loss_all / len(train_indices)
        val_loss = val_loss_all / len(val_indices)
        logger.info("Epoch: %d, LR: %.5f, Ratio: %.2f, Train Loss: %.3f, Val Loss: %.3f" % (epoch + 1, lr, config.explainers.param.sparsity, train_loss, val_loss))
    refine_saving_path = os.path.join(result_dir, config.explainers.explainer_saving_name)
    torch.save(explainer, refine_saving_path)
    explainer = torch.load(refine_saving_path)

    x_collector = XCollector()
    if config.models.param.graph_classification:
    
        for i, data in enumerate(test_loader):
            data = data.to(device)
            if data.edge_attr is None:
                data.edge_attr = torch.ones(data.edge_index.shape[1], 1).to(device)

            prediction = model(data).argmax(-1).item()
            if prediction != data.y.item():
                continue
            logger.info(f"explaining example {test_indices[i]}.")
            logger.info(f"prediction: {prediction} | true label: {data.y.item()}")
            
            related_preds = {}

            if config.datasets.ground_truth_available:
                _max_ex_size, _num_motifs = choose_explainer_param(data, dataset)
                max_ex_size = _max_ex_size if config.explainers.param.max_ex_size == -1 else config.explainers.param.max_ex_size
                num_motifs = _num_motifs if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                sparsity = max_ex_size / data.num_nodes if config.explainers.param.sparsity == -1 else config.explainers.param.sparsity
            else:
                max_ex_size = config.explainers.param.max_ex_size
                num_motifs = 1 if config.experiments.num_motifs == -1 else config.experiments.num_motifs
                if max_ex_size < 1:
                    max_ex_size = max(3, int(max_ex_size * data.num_nodes))
                sparsity = max_ex_size / data.num_nodes if config.explainers.param.sparsity == -1 else config.explainers.param.sparsity
 
            start_time = time.time()
            # if len(data.edge_attr.shape) == 1:
            #     data.edge_attr = data.edge_attr.unsqueeze(1)
            edge_mask = explainer.explain_graph(data, ratio=sparsity,
                                                    lr=config.explainers.param.lr, epoch=config.explainers.param.epoch)
            end_time = time.time()

            related_preds['running_time'] = end_time - start_time
            related_preds['test_idx'] = test_indices[i]
            related_preds['test_label'] = data.y.item()
            related_preds['test_pred'] = prediction
            expl_graph = None
            if config.datasets.ground_truth_available:
                gt_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
            else:
                gt_edge_mask = None
                
            predict_fn = get_reward_func_for_gnn_gc(model, prediction, payoff_type='prob')
            stats, other_info = compute_explanation_stats(data, gt_edge_mask, edge_mask=edge_mask,
                                                          num_motifs=num_motifs, max_nodes=max_ex_size,
                                                          predict_fn=predict_fn,
                                                          subgraph_building_method=config.datasets.subgraph_building_method)
            for key, value in stats.items():
                related_preds[key] = value
            expl_graph = other_info['expl_graph']
            logger.info(f"explanation: {expl_graph.nodes}")
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

                graph = to_networkx(data, to_undirected=True)
                plot_utils.plot(
                    graph,
                    nodelist=[list(expl_graph.nodes)],
                    expl_graph=expl_graph,
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

    
if __name__ == '__main__':
    pass
