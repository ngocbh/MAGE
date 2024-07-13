import torch

from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    def __init__(self, masks, masker, *masker_args, **masker_kwargs):
        super().__init__()
        self.masks = masks
        self.masker = masker
        self.masker_args = masker_args
        self.masker_kwargs = masker_kwargs

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        masked_input = self.masker(self.masks[idx],
                                   *self.masker_args,
                                   **self.masker_kwargs)
        return masked_input
