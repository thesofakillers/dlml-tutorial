"""
Discretized Logistic Mixture Likelihood training and sampling

Very barebones code, made to accompany blogpost at
https://www.giuliostarace.com/posts/dlml-tutorial/

Largely inspired by:
https://github.com/lukashermann/hulc
https://github.com/Rayhane-mamah/Tacotron-2
https://github.com/openai/pixel-cnn
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_dlml_loss(
    means,
    log_scales,
    mixture_logits,
    y,
    output_min_bound,
    output_max_bound,
    num_y_vals,
    reduction="mean",
):
    """
    Computes the Discretized Logistic Mixture Likelihood loss
    """
    inv_scales = torch.exp(-log_scales)

    y_range = output_max_bound - output_min_bound
    # explained in text
    epsilon = (0.5 * y_range) / (num_y_vals - 1)
    # convenience variable
    centered_y = y.unsqueeze(-1).repeat(1, 1, means.shape[-1]) - means
    # inputs to our sigmoid functions
    upper_bound_in = inv_scales * (centered_y + epsilon)
    lower_bound_in = inv_scales * (centered_y - epsilon)
    # remember: cdf of logistic distr is sigmoid of above input format
    upper_cdf = torch.sigmoid(upper_bound_in)
    lower_cdf = torch.sigmoid(lower_bound_in)
    # finally, the probability mass and equivalent log prob
    prob_mass = upper_cdf - lower_cdf
    vanilla_log_prob = torch.log(torch.clamp(prob_mass, min=1e-12))

    # edges
    low_bound_log_prob = upper_bound_in - F.softplus(
        upper_bound_in
    )  # log probability for edge case of 0 (before scaling)
    upp_bound_log_prob = -F.softplus(
        lower_bound_in
    )  # log probability for edge case of 255 (before scaling)
    # middle
    mid_in = inv_scales * centered_y
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    log_prob_mid = log_pdf_mid - np.log((num_y_vals - 1) / 2)

    # Create a tensor with the same shape as 'y', filled with zeros
    log_probs = torch.zeros_like(y)
    # conditions for filling in tensor
    is_near_min = y < output_min_bound + 1e-3
    is_near_max = y > output_max_bound - 1e-3
    is_prob_mass_sufficient = prob_mass > 1e-5
    # And then fill it in accordingly
    # lower edge
    log_probs[is_near_min] = low_bound_log_prob[is_near_min]
    # upper edge
    log_probs[is_near_max] = upp_bound_log_prob[is_near_max]
    # vanilla case
    log_probs[~is_near_min & ~is_near_max & is_prob_mass_sufficient] = vanilla_log_prob[
        ~is_near_min & ~is_near_max & is_prob_mass_sufficient
    ]
    # extreme case where prob mass is too small
    log_probs[~is_near_min & ~is_near_max & ~is_prob_mass_sufficient] = log_prob_mid[
        ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
    ]

    # modeling which mixture to sample from
    log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1)

    # log likelihood
    log_likelihood = torch.sum(torch.logsumexp(log_probs), dim=-1)

    # loss is just negative log likelihood
    loss = -log_likelihood

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)


def train_single_batch(
    model, batch, optimizer, output_min_bound, output_max_bound, num_y_vals
):
    """
    Trains a single batch of data.
    """
    # each of these have shape (B x out_dim x K)
    means, log_scales, mixture_logits = model(**batch["inputs"])
    # shape (B x out_dim)
    y = batch["targets"]

    optimizer.zero_grad()
    loss = compute_dlml_loss(
        means,
        log_scales,
        mixture_logits,
        y,
        output_min_bound,
        output_max_bound,
        num_y_vals,
    )
    loss.backwards()
    optimizer.step()


def sample(model, batch):
    """
    Samples from our model
    """
    means, log_scales, mixture_logits = model(**batch["inputs"])

    r1, r2 = 1e-5, 1.0 - 1e-5
    temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
    temp = mixture_logits - torch.log(-torch.log(temp))
    argmax = torch.argmax(temp, -1)

    # number of distributions in mixture
    k = means.shape[-1]
    # (K dimensional vector, e.g. [0 0 0 1 0 0 0 0] for K=8, argmax=3
    dist_one_hot = torch.eye(k)[argmax]

    # use it to sample, and aggregate over the batch
    sampled_log_scale = (dist_one_hot * log_scales).sum(dim=-1)
    sampled_mean = (dist_one_hot * means).sum(dim=-1)

    # scale the (0,1) uniform distribution and re-center it
    y = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2

    sampled_output = sampled_mean + torch.exp(sampled_log_scale) * (
        torch.log(y) - torch.log(1 - y)
    )

    return sampled_output
