import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from dig.xgraph.dataset import MoleculeDataset, SynGraphDataset, SentiGraphDataset, BA_LRP
from dataset.mutag0 import Mutag0
from dataset.ba_house_grid import BaHouseGrid
from dataset.benzene import Benzene
from dataset.spmotif import SPMotif
from dataset.mnist_bin import MNIST75sp_Binary
from dataset.mnist_superpixel import MNISTSuperpixels
from torch import default_generator
import torch_geometric.transforms as T
import os


def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() == 'mutag0':
        return Mutag0(root=dataset_root, name=dataset_name)
    elif 'spmotif' in dataset_name.lower():
        return SPMotif(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() == 'benzene':
        return Benzene(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() == 'mnist75sp':
        return MNIST75sp_Binary(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'twitter']:
        return SentiGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(BaHouseGrid.names.keys()):
        return BaHouseGrid(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(SynGraphDataset.names.keys()):
        return SynGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ['ba_lrp']:
        return BA_LRP(root=dataset_root)
    elif dataset_name.lower() == 'mnist_superpixel':
        transform = T.Cartesian(cat=False, max_value=9)
        return MNISTSuperpixels(os.path.join(dataset_root, dataset_name), True, transform=transform)
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=2, num_workers=2):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        from functools import partial
        generator = torch.Generator()
        generator.manual_seed(seed)
        train, eval, test = random_split(dataset,
                                         lengths=[num_train, num_eval, num_test],
                                         generator=generator)

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader
