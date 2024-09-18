"""
This script contains utility functions used in calculating contrastive loss. 
Some are adapted from the following script in the contrastive-mir-learning repository 
(corresponding to reference [18] in the paper):
https://github.com/andrebola/contrastive-mir-learning/blob/master/utils.py
"""

import torch
import numpy as np
from functools import partial


def get_weight_mask(N, pos_1_inds, pos_2_inds, weights):
    """
    Creates weight mask of contrastive pairs based on provided indices
    and weights.

    Args:
        N: number of pairs in batch
        pos_1_inds: indices of first item in each pair
        pos_2_inds: indices of second item in each pair
        weights: weight to be assigned to each pair
    """
    weights_mask = torch.zeros((2 * N, 2 * N))
    weights_mask[pos_1_inds, pos_2_inds] = weights
    weights_mask[pos_2_inds, pos_1_inds] = weights
    return weights_mask


def get_weight_mask_a_im(N):
    """
    Gets weight mask in simple A_IM case, where positive examples are
    only the pairs provided, and weights are all 1.
    """
    pos_1_inds = list(range(N))
    pos_2_inds = list(range(N, 2 * N))
    weights = torch.ones((N))
    return get_weight_mask(N, pos_1_inds, pos_2_inds, weights)


def get_masked_scores(s, weights_mask, denom):
    """
    Calculate contrastive loss based on provided similarities and weights.

    Args:
        s: similarity matrix (2n x 2n)
        weights_mask: mask of weights (2n x 2n), i.e. the alpha_{i,j}
        denom: denominator in NT-XEnt loss function

    Returns:
        Loss value
    """
    # Scale weights
    weights_mask_scaled = weights_mask / weights_mask.sum(dim=-1)
    s_masked = torch.log(s + 1e-5) * weights_mask_scaled.cuda()
    # Divide by denominator (simplified by noting that log(a/b)=log(a)-log(b))
    return (s_masked.sum(dim=-1) - torch.log(denom)).neg().mean()


def embeddings_to_cosine_similarity_matrix(z):
    """Converts a a tensor of n embeddings to an (n, n) tensor of similarities."""
    cosine_similarity = torch.matmul(z, z.t())
    embedding_norms = torch.norm(z, p=2, dim=1)
    embedding_norms_mat = embedding_norms.unsqueeze(0) * embedding_norms.unsqueeze(1)
    cosine_similarity = cosine_similarity / torch.maximum(
        embedding_norms_mat, torch.tensor(1e-8)
    )
    return cosine_similarity


def contrastive_loss(model_out, t=1, weight_mask=None):
    """
    Calculates contrastive loss given model outputs.

    Args:
        model_out: Tuple containing (anchor outputs, positive outputs)
        t: contrastive temperature
        weight_mask: weight mask (2n x 2n), i.e. the alpha_{i,j}

    Returns:
        Loss value
    """
    z = torch.cat(model_out, dim=0)

    s = embeddings_to_cosine_similarity_matrix(z)
    N = int(s.shape[0] / 2)
    s = torch.exp(s / t)
    try:
        s = s * (1 - torch.eye(len(s), len(s)).cuda())
    except AssertionError:
        s = s * (1 - torch.eye(len(s), len(s)))
    denom = s.sum(dim=-1)

    # If no weight mask is provided, default to standard approach
    if weight_mask is None:
        weight_mask = get_weight_mask_a_im(N)

    return get_masked_scores(s, weight_mask, denom)


def get_loss_function(loss_params):
    return partial(contrastive_loss, **{"t": loss_params["contrastive_temperature"]})


def get_average_losses(losses_lst):
    """
    Utility function for averaging all losses from epoch to get dictionary containing
    loss values.
    """
    epoch_losses = {k: [] for k in losses_lst[0]}
    for losses in losses_lst:
        for k, loss in losses.items():
            try:
                epoch_losses[k].append(loss.cpu().detach().item())
            except:
                epoch_losses[k].append(loss)
    epoch_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
    return epoch_losses
