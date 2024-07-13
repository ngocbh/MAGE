import numpy as np
import os
import numbers
import collections

import optuna
from sklearn.model_selection import ParameterGrid
import logging 
import sys
from utils import get_logger
optuna.logging.set_verbosity(optuna.logging.DEBUG)

def dummy(pipeline, config, parent=True):
    if parent:
        config.result_dir = os.path.join(config.result_dir, 'dummy')
    config.record_filename = os.path.join(config.result_dir, 'result.json')
    pipeline(config)

    
def single_motif(pipeline, config, parent=True):
    assert config.experiments.num_motifs == 1
    dataset_name = config.datasets.dataset_name
    explainer_name = config.explainers.explainer_name

    if parent:
        config.result_dir = os.path.join(config.result_dir, config.experiments.name)
    config.record_filename = os.path.join(config.result_dir, 'result.json')
    if not os.path.isdir(config.result_dir):
        os.makedirs(config.result_dir, exist_ok=True)

    sparsity_param = config.experiments.sparsity_param[explainer_name]
    explanation_sparsity = config.experiments.explanation_sparsity[dataset_name]
    config.explainers.param[sparsity_param] = explanation_sparsity
    pipeline(config)


def multi_motifs(pipeline, config, parent=True):
    explainer_name = config.explainers.explainer_name
    sparsity_param = config.experiments.sparsity_param[explainer_name]

    if parent:
        config.result_dir = os.path.join(config.result_dir, config.experiments.name)
    config.record_filename = os.path.join(config.result_dir, 'result.json')
    if not os.path.isdir(config.result_dir):
        os.makedirs(config.result_dir, exist_ok=True)

    config.explainers.param[sparsity_param] = -1
    pipeline(config)