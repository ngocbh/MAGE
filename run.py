import hydra
import numpy as np
import os
import torch

import rmage
import experiments

from baselines import subgraphx
from baselines import gnn_explainer
from baselines import pgexplainer_edges
from baselines import gstarx
from baselines import grad_cam
from baselines import match_explainer
from baselines import refine
from baselines import same
from utils import fix_random_seed

pipelines = {
    'subgraphx': subgraphx.pipeline,
    'mage': rmage.pipeline,
    'gnn_explainer': gnn_explainer.pipeline,
    'pgexplainer': pgexplainer_edges.pipeline,
    'gstarx': gstarx.pipeline,
    'grad_cam': grad_cam.pipeline,
    'match_explainer': match_explainer.pipeline,
    'refine': refine.pipeline,
    'same': same.pipeline
}

experiments = {
    'dummy': experiments.dummy,
    'single_motif': experiments.single_motif,
    'multi_motifs': experiments.multi_motifs,
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    fix_random_seed(config.random_seed)
    cwd = os.path.dirname(os.path.abspath(__file__))

    config.datasets.dataset_root = os.path.join(cwd, "datasets")
    config.models.gnn_saving_dir = os.path.join(cwd, "checkpoints")
    config.explainers.explanation_result_dir = os.path.join(cwd, f"results/{config.run_id}")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    result_dir = os.path.join(config.explainers.explanation_result_dir,
                              config.datasets.dataset_name,
                              config.models.gnn_name)

    config.result_dir = result_dir
    config.record_filename = os.path.join(config.result_dir, 'result.json')
    if not os.path.isdir(config.result_dir):
        os.makedirs(config.result_dir, exist_ok=True)

    pipeline = pipelines[config.explainers.explainer_name]
    
    expt = experiments[config.experiments.name]
    expt(pipeline, config)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
