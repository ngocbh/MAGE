import skimage
import math
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from shap.maskers import Masker

from mage.utils.validation import check_mask_input


def get_default_transformer():
    return torchvision.transforms.ToTensor()


def patch_segmentation(input, patch_size=(32, 32)):
    w, h, c = input.shape
    segments = np.zeros((w, h), dtype=int)
    ppw = int(math.ceil(w / patch_size[0]))
    for i in range(w):
        for j in range(h):
            segments[i, j] = (i//patch_size[0]) * ppw + j//patch_size[1]
    return segments


class ImageMasker(Masker):
    def __init__(self, transformer=None, 
                 overseg_method='slic',
                 rag_method='mean_color',
                 *args, **kwargs):
        self.transformer = transformer if transformer is not None else get_default_transformer()
        self.overseg_method = overseg_method
        self.rag_method = rag_method
        self.overseg_args = args
        self.overseg_kwargs = kwargs

        super(ImageMasker, self).__init__()

    def __call__(self, mask, input, segments, segment_indices, baseline):
        """
            input: H x W x C
        """
        num_segments = len(segment_indices)
        if isinstance(mask, np.ndarray):
            if isinstance(mask[0], frozenset):
                mask = [list(x) for x in mask]
        mask = check_mask_input(mask, size=num_segments)
        masked_input = input.copy()
        zeros = np.where(mask == 0.)[0]
        for sp_idx in zeros:
            sp_mask = (segments == sp_idx)
            masked_input[sp_mask] = baseline

        return self.transformer(masked_input)

    def over_segment(self, input):
        if self.overseg_method == "slic":
            segments = skimage.segmentation.slic(input,
                                                 *self.overseg_args,
                                                 start_label=0,
                                                 **self.overseg_kwargs)
        elif self.overseg_method == "felzen":
            segments = skimage.segmentation.felzenszwalb(input,
                                                         *self.overseg_args,
                                                         **self.overseg_kwargs)
        elif self.overseg_method == "quick":
            segments = skimage.segmentation.quickshift(input,
                                                       *self.overseg_args,
                                                       **self.overseg_kwargs)
        elif self.overseg_method == "water":
            gradient = skimage.filters.sobel(skimage.color.rgb2gray(input))
            segments = skimage.segmentation.watershed(gradient,
                                                      *self.overseg_args,
                                                      **self.overseg_kwargs)
        elif self.overseg_method == "patch":
            segments = patch_segmentation(input,
                                          *self.overseg_args,
                                          **self.overseg_kwargs)
        else:
            raise ValueError("unknown method")

        return segments

    def build_rag(self, input):
        segments = self.over_segment(input)
        if self.rag_method == 'boundary':
            edges = skimage.filters.sobel(skimage.color.rgb2gray(input))
            graph = skimage.graph.rag_boundary(segments, edges)
        elif self.rag_method == 'mean_color':
            graph = skimage.graph.rag_mean_color(input, segments)
        else:
            raise ValueError("unknown rag building method.")
        return graph, segments
