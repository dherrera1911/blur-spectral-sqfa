"""Auxiliary functions for the spectral ideal observer fitting."""

import torch
from sqfa.model import SecondMomentsSQFA

class SQFABlock(SecondMomentsSQFA):
    """
    Second-moments SQFA model, but that takes as inputs
    scatter matrices with block structure for memory
    and computational efficiency.
    """
    def __init__(self, n_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_blocks = n_blocks

    def transform_scatters(self, data_scatters):
        """
        Transform data scatter matrices with block structure to feature
        space scatter matrices.

        Parameters
        ----------
        data_scatters : torch.Tensor
            Tensor of shape (n_classes, n_blocks, block_dim, block_dim), with second
            moment or covariance matrices.

        Returns
        -------
        torch.Tensor shape (n_classes, n_filters, n_filters)
            Covariances of the transformed features.
        """
        n_filters = self.filters.shape[0]
        block_dim = self.filters.shape[-1] // self.n_blocks

        filters_block = self.filters.view(n_filters, self.n_blocks, block_dim)
        feature_scatters = torch.einsum(
          "ijk,...jkl,mjl->...im", filters_block, data_scatters, filters_block)
        return feature_scatters

