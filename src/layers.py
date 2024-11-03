# Parts of this code are adapted from:
#   - https://github.com/microsoft/unilm/tree/master/Diff-Transformer
#   - https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

import math
import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func


def MultiheadAttn(
    q: torch.tensor, k: torch.tensor, v: torch.tensor, causal: bool = True
):

    _, _, N, head_dim = q.size()

    sm_scale = 1 / math.sqrt(head_dim)

    M = torch.tril(torch.ones((N, N), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if causal:
        p[:, :, M == 0] = float("-inf")

    p = torch.softmax(p.float(), dim=-1).half()

    attn = torch.matmul(p, v)

    return attn


def MultiheadDiffAttn(
    q1: torch.tensor,
    q2: torch.tensor,
    k1: torch.tensor,
    k2: torch.tensor,
    v: torch.tensor,
    lambda_scale: float = 0.5,
    causal: bool = True,
):

    _, _, N, head_dim = q1.size()

    sm_scale = 1 / math.sqrt(head_dim)

    M = torch.tril(torch.ones((N, N), device="cuda"))

    # First softmax
    p1 = torch.matmul(q1, k1.transpose(2, 3)) * sm_scale

    if causal:
        p1[:, :, M == 0] = float("-inf")

    p1 = torch.softmax(p1.float(), dim=-1).half()

    # Second softmax
    p2 = torch.matmul(q2, k2.transpose(2, 3)) * sm_scale

    if causal:
        p2[:, :, M == 0] = float("-inf")

    p2 = torch.softmax(p2.float(), dim=-1).half()

    p = p1 - lambda_scale * p2

    attn = torch.matmul(p, v)

    return attn


def MultiheadFlashAttn(
    q: torch.tensor, k: torch.tensor, v: torch.tensor, causal: bool = True
):

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn = flash_attn_func(q, k, v, causal=causal)

    attn = attn.transpose(1, 2)

    return attn


def MultiheadFlashDiffAttn(
    q1: torch.tensor,
    q2: torch.tensor,
    k1: torch.tensor,
    k2: torch.tensor,
    v: torch.tensor,
    lambda_scale: float = 0.5,
    causal: bool = True,
):
    B, num_heads, N, head_dim = q1.size()

    v = v.view(B, num_heads, N, 2, head_dim)

    v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

    q1 = q1.transpose(1, 2)
    q2 = q2.transpose(1, 2)

    k1 = k1.transpose(1, 2)
    k2 = k2.transpose(1, 2)

    v1 = v1.transpose(1, 2)
    v2 = v2.transpose(1, 2)

    attn11 = flash_attn_func(q1, k1, v1, causal=causal)
    attn12 = flash_attn_func(q1, k1, v2, causal=causal)
    attn1 = torch.cat([attn11, attn12], dim=-1)

    attn21 = flash_attn_func(q2, k2, v1, causal=causal)
    attn22 = flash_attn_func(q2, k2, v2, causal=causal)
    attn2 = torch.cat([attn21, attn22], dim=-1)

    attn = attn1 - lambda_scale * attn2

    attn = attn.transpose(1, 2)

    return attn
