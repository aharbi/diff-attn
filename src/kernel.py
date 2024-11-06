"""
Differential Attention Kernel. Modified version of Triton's fused attention kernel example.

Credits:
* Triton's implementation: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
* Flash attention papers (https://arxiv.org/abs/2205.14135,
                          https://tridao.me/publications/flash2/flash2.pdf)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import torch

import triton
import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64]
    for BN in [32]
    for s in ([1] if is_hip() else [3, 4, 7])
    for w in [4]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.jit
def _attn_fwd(
    # Pointers
    Q1,
    Q2,
    K1,
    K2,
    V,
    sm_scale,
    M1,
    M2,
    Out,  #
    # Strides
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    # Shapes
    Z,
    H,
    N_CTX,  #
    HEAD_DIM_Q: tl.constexpr,  #
    HEAD_DIM_K: tl.constexpr,  #
    HEAD_DIM_V: tl.constexpr,  #
    # Constants
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM_V)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H  # Batch index we are processing
    off_h = off_hz % H  # Head index we are processing

    # Get the offest of the specific batch element and head number combination
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

    # Q Block pointers
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + qvk_offset,
        shape=(N_CTX, HEAD_DIM_Q),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_Q),
        order=(1, 0),
    )

    Q2_block_ptr = tl.make_block_ptr(
        base=Q2 + qvk_offset,
        shape=(N_CTX, HEAD_DIM_Q),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_Q),
        order=(1, 0),
    )

    # K Block pointers
    K1_block_ptr = tl.make_block_ptr(
        base=K1 + qvk_offset,
        shape=(HEAD_DIM_K, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_K, BLOCK_N),
        order=(0, 1),  # NOTE: Transposed K I guess?
    )

    K2_block_ptr = tl.make_block_ptr(
        base=K2 + qvk_offset,
        shape=(HEAD_DIM_K, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_K, BLOCK_N),
        order=(0, 1),  # NOTE: Transposed K I guess?
    )

    # V Block pointers
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V),
        order=v_order,
    )

    # O Block pointers
    O_block_ptr = tl.make_block_ptr(
        base=Out + v_offset,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # NOTE: Used for masking memory addresses beyond the head dimension
    offs_n = tl.arange(0, BLOCK_N)

    # NOTE: Initialized in SRAM
    # initialize pointer to m and l
    m_i_1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max for each row
    l_i_1 = (
        tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    )  # Normalization factor for each row

    m_i_2 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max for each row
    l_i_2 = (
        tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    )  # Normalization factor for each row

    acc_1 = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q1 = tl.load(Q1_block_ptr)
    q2 = tl.load(Q2_block_ptr)

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc_1, l_i_1, m_i_1 = _attn_fwd_inner(
            acc_1,
            l_i_1,
            m_i_1,
            q1,
            K1_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM_V,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
        )

        acc_2, l_i_2, m_i_2 = _attn_fwd_inner(
            acc_2,
            l_i_2,
            m_i_2,
            q2,
            K2_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM_V,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc_1, l_i_1, m_i_1 = _attn_fwd_inner(
            acc_1,
            l_i_1,
            m_i_1,
            q1,
            K1_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM_V,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
        )

        acc_2, l_i_2, m_i_2 = _attn_fwd_inner(
            acc_2,
            l_i_2,
            m_i_2,
            q2,
            K2_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM_V,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
        )

    # epilogue
    m_i_1 += tl.math.log2(l_i_1)
    m_i_2 += tl.math.log2(l_i_2)

    acc = acc_1 / l_i_1[:, None] - acc_2 / l_i_2[:, None]

    m_ptrs_1 = M1 + off_hz * N_CTX + offs_m
    m_ptrs_2 = M2 + off_hz * N_CTX + offs_m

    tl.store(m_ptrs_1, m_i_1)
    tl.store(m_ptrs_2, m_i_2)

    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _diff_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q1, q2, k1, k2, v, causal, sm_scale):
        BLOCK_M = 16
        BLOCK_N = 16

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q1.shape[-1], k1.shape[-1]

        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]

        # NOTE: For DA, we probably need to remove the second assertion
        assert HEAD_DIM_Q == HEAD_DIM_K  # and HEAD_DIM_K == HEAD_DIM_V

        # NOTE: This is fine
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(v)

        stage = 3 if causal else 1
        extra_kern_args = {}

        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # NOTE: Axis 0: Over tokens. Axis 1: Over Batch_size * num_heads
        grid = lambda args: (
            triton.cdiv(q1.shape[2], args["BLOCK_M"]),
            q1.shape[0] * q1.shape[1],
            1,
        )

        # NOTE: Maximum value
        M_1 = torch.empty(
            (q1.shape[0], q1.shape[1], q1.shape[2]),
            device=q1.device,
            dtype=torch.float32,
        )

        M_2 = torch.empty(
            (q1.shape[0], q1.shape[1], q1.shape[2]),
            device=q1.device,
            dtype=torch.float32,
        )

        _attn_fwd[grid](
            q1,
            q2,
            k1,
            k2,
            v,
            sm_scale,
            M_1,
            M_2,
            o,  #
            q1.stride(0),
            q1.stride(1),
            q1.stride(2),
            q1.stride(3),  #
            k1.stride(0),
            k1.stride(1),
            k1.stride(2),
            k1.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q1.shape[0],
            q1.shape[1],  #
            N_CTX=q1.shape[2],  #
            HEAD_DIM_Q=HEAD_DIM_Q,  #
            HEAD_DIM_K=HEAD_DIM_K,  #
            HEAD_DIM_V=HEAD_DIM_V,  #
            STAGE=stage,  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            **extra_kern_args,
        )

        return o


if __name__ == "__main__":
    # Quick test
    import math

    from layers import MultiheadDiffAttn, MultiheadDiffAttnKernel

    B = 64
    H = 8

    q1 = torch.rand(B, H, 1024, 32 // 2, dtype=torch.float16).to("cuda")
    q2 = torch.zeros(B, H, 1024, 32 // 2, dtype=torch.float16).to("cuda")
    k1 = torch.rand(B, H, 1024, 32 // 2, dtype=torch.float16).to("cuda")
    k2 = torch.zeros(B, H, 1024, 32 // 2, dtype=torch.float16).to("cuda")
    v = torch.rand(B, H, 1024, 32, dtype=torch.float16).to("cuda")

    y1 = MultiheadDiffAttnKernel(q1, q2, k1, k2, v, causal=False, lambda_scale=1)
    y2 = MultiheadDiffAttn(q1, q2, k1, k2, v, causal=False, lambda_scale=1)

    print(y1 - y2)

    print(torch.allclose(y1, y2, atol=1e-2))
