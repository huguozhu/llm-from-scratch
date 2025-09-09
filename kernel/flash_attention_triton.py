import triton
import triton.language as tl
import torch
from torch import Tensor
from jaxtyping import Float
import math
import einx


@triton.jit
def flash_attention_forward_kernel(
    q,
    k,
    v,
    o,
    l,
    stride_qb,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_on,
    stride_od,
    stride_lb,
    stride_ln,
    n: tl.int32,
    d_scale: tl.float32,
    IS_CAUSAL: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_tq = tl.program_id(1)

    q_block_ptr = tl.make_block_ptr(
        base=q + pid_b * stride_qb,
        shape=(n, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_tq * BQ, 0),
        block_shape=(BQ, D),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k + pid_b * stride_kb,
        shape=(D, n),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BK),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v + pid_b * stride_vb,
        shape=(n, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BK, D),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o + pid_b * stride_ob,
        shape=(n, D),
        strides=(stride_on, stride_od),
        offsets=(pid_tq * BQ, 0),
        block_shape=(BQ, D),
        order=(1, 0),
    )
    l_ptrs = l + pid_b * stride_lb + (pid_tq * BQ + tl.arange(0, BQ))

    m_i = tl.full([BQ], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BQ], dtype=tl.float32)
    o_i = tl.zeros([BQ, D], dtype=tl.float32)

    q_i = tl.load(q_block_ptr, boundary_check=(0, 1))
    q_i *= d_scale

    j = 0

    loop_end = tl.cdiv(n, BK)
    if IS_CAUSAL:
        loop_end = tl.cdiv((pid_tq + 1) * BQ, BK)

    for j in range(loop_end):
        k_j = tl.load(k_block_ptr, boundary_check=(0, 1))
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1))

        s_ij = tl.dot(q_i, k_j)

        if IS_CAUSAL:
            offs_q = pid_tq * BQ + tl.arange(0, BQ)
            offs_k = j * BK + tl.arange(0, BK)
            s_ij += tl.where(offs_q[:, None] >= offs_k[None, :], 0, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        scale = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])

        l_new = scale * l_i + tl.sum(p_ij, axis=1)
        o_i = scale[:, None] * o_i + tl.dot(p_ij.to(v_j.dtype), v_j)

        l_i = l_new
        m_i = m_new

        k_block_ptr = tl.advance(k_block_ptr, (0, BK))
        v_block_ptr = tl.advance(v_block_ptr, (BK, 0))

    o_i /= l_i[:, None]
    l_i = m_i + tl.log(l_i + eps)

    tl.store(o_block_ptr, o_i.to(o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(l_ptrs, l_i, mask=(pid_tq * BQ + tl.arange(0, BQ)) < n)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Float[Tensor, "b n d"],
        k: Float[Tensor, "b n d"],
        v: Float[Tensor, "b n d"],
        is_causal=False,
        eps=1e-6,
    ):
        device = q.device
        b, n, d = q.shape
        scale = 1.0 / math.sqrt(d)

        o = torch.empty_like(q)
        l = torch.empty((b, n), dtype=torch.float32, device=device)

        BQ = 64
        BK = 64

        # assert d in [16, 32, 64, 128]

        grid = (b, triton.cdiv(n, BQ))

        flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            o,
            l,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            n=n,
            d_scale=scale,
            IS_CAUSAL=is_causal,
            BQ=BQ,
            BK=BK,
            D=d,
            eps=eps,
        )
        ctx.save_for_backward(q, k, v, l, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, *grad_out):
        q, k, v, l, o = ctx.saved_tensors
        is_causal = ctx.is_causal
        b, n, d = q.shape
        scale = 1.0 / math.sqrt(d)

        # Use torch.compile for speed
        dQ, dK, dV = _flash_attn_backward_compiled(
            q, k, v, o, grad_out[0], l, scale, is_causal
        )
        return dQ, dK, dV, None


@torch.compile
def _flash_attn_backward_compiled(q, k, v, o, do, l, scale, is_causal):
    b, n, d = q.shape
    m = k.shape[1]

    s = einx.dot("b q d, b k d -> b q k", q, k) * scale  # (b, q_len, k_len)

    if is_causal:
        # Upper triangular (excluding diagonal)
        causal_mask = torch.ones_like(s, dtype=torch.bool).triu_(diagonal=1)
        s = s.masked_fill(causal_mask, float("-inf"))

    s_max = s.amax(dim=-1, keepdim=True).detach()  # stable max
    p_unmasked = (s - s_max).exp()
    p_sum = p_unmasked.sum(dim=-1, keepdim=True)

    p = p_unmasked / (p_sum + 1e-10)  # (b, q_len, k_len)

    if is_causal:
        p = p.masked_fill(causal_mask, 0.0)

    dv = einx.dot("b q k, b q d -> b k d", p, do)  # (b, k_len, d)

    dp = einx.dot("b q d, b k d -> b q k", do, v)  # (b, q_len, k_len)

    if is_causal:
        dp = dp.masked_fill(causal_mask, 0.0)

    p_dot_dp = torch.einsum("b q k, b q k -> b q", p, dp).unsqueeze(-1)  # (b, q, 1)
    ds = p * (dp - p_dot_dp)  # (b, q, k)

    dq = einx.dot("b q k, b k d -> b q d", ds, k) * scale  # (b, q_len, d)
    dk = einx.dot("b q k, b q d -> b k d", ds, q) * scale  # (b, k_len, d)

    return dq, dk, dv
