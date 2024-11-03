# Parts of this code are adapted from: https://github.com/microsoft/unilm/tree/master/Diff-Transformer
import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func


def MultiheadAttn(q, k, v, embed_dim, num_heads, causal=True):
    head_dim = embed_dim // num_heads

    B, N, _ = q.size()

    q = q.view(B, N, num_heads, head_dim)
    k = k.view(B, N, num_heads, head_dim)
    v = v.view(B, N, num_heads, head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q *= head_dim**-0.5

    attn_weights = torch.matmul(q, k.transpose(-1, -2))

    if causal:
        attn_mask = torch.triu(
            torch.zeros([N, N]).float().fill_(float("-inf")).type_as(attn_weights),
            1,
        )

    attn_weights += attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn = torch.matmul(attn_weights, v)
    attn = attn.transpose(1, 2).reshape(B, N, num_heads * head_dim)

    return attn


def MultiheadDiffAttn(q, k, v, embed_dim, num_heads, lambda_full=0.5, causal=True):
    head_dim = embed_dim // num_heads // 2

    B, N, _ = q.size()

    q = q.view(B, N, 2 * num_heads, head_dim)
    k = k.view(B, N, 2 * num_heads, head_dim)
    v = v.view(B, N, num_heads, 2 * head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q *= head_dim**-0.5

    attn_weights = torch.matmul(q, k.transpose(-1, -2))

    if causal:
        attn_mask = torch.triu(
            torch.zeros([N, N]).float().fill_(float("-inf")).type_as(attn_weights),
            1,
        )

    attn_weights += attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.view(B, num_heads, 2, N, N)
    attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

    attn = torch.matmul(attn_weights, v)
    attn = attn.transpose(1, 2).reshape(B, N, num_heads * 2 * head_dim)

    return attn


def MultiheadFlashAttn(q, k, v, embed_dim, num_heads, causal=True):
    head_dim = embed_dim // num_heads

    B, N, _ = q.size()

    q = q.view(B, N, num_heads, head_dim)
    k = k.view(B, N, num_heads, head_dim)
    v = v.view(B, N, num_heads, head_dim)

    attn = flash_attn_func(q, k, v, causal=causal)

    attn = attn.reshape(B, N, num_heads * head_dim)

    return attn


def MultiheadFlashDiffAttn(q, k, v, embed_dim, num_heads, lambda_full=0.5, causal=True):
    head_dim = embed_dim // num_heads // 2

    B, N, _ = q.size()

    q = q.view(B, N, 2 * num_heads, head_dim)
    k = k.view(B, N, 2 * num_heads, head_dim)
    v = v.view(B, N, num_heads, 2, head_dim)

    q = q.reshape(B, N, num_heads, 2, head_dim)
    k = k.reshape(B, N, num_heads, 2, head_dim)
    q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
    k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
    v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

    attn11 = flash_attn_func(q1, k1, v1, causal=causal)
    attn12 = flash_attn_func(q1, k1, v2, causal=causal)
    attn1 = torch.cat([attn11, attn12], dim=-1)

    attn21 = flash_attn_func(q2, k2, v1, causal=causal)
    attn22 = flash_attn_func(q2, k2, v2, causal=causal)
    attn2 = torch.cat([attn21, attn22], dim=-1)

    attn = attn1 - lambda_full * attn2
    attn = attn.reshape(B, N, num_heads * 2 * head_dim)

    return attn
