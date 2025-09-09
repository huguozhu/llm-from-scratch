import torch
from torch import Tensor
from jaxtyping import Float
import math
import einx


class FlashAttentionMock(torch.autograd.Function):
    @staticmethod
    def forward_naive(
        ctx,
        q: Float[Tensor, "... n d"],
        k: Float[Tensor, "... n d"],
        v: Float[Tensor, "... n d"],
        is_causal: bool,
    ):
        """
        Args:
            q (Tensor): [... n, d]
            k (Tensor): [... n, d]
            v (Tensor): [... n, d]
            is_causal (bool):
        """

        d = q.shape[-1]
        p = einx.dot("... n d, ... m d -> ... n m", q, k) / math.sqrt(d)
        o = torch.nn.functional.softmax(p, dim=-1) @ v
        l = torch.logsumexp(p, dim=-1)
        ctx.save_for_backward(q, k, v, l, o)
        return o

    @staticmethod
    def forward(
        ctx,
        q: Float[Tensor, "... n d"],
        k: Float[Tensor, "... n d"],
        v: Float[Tensor, "... n d"],
        is_causal: bool,
        naive=False,
        bq=16,
        bk=16,
        eps=1e-6,
    ):
        if naive:
            return FlashAttentionMock.forward_naive(ctx, q, k, v, is_causal)

        # support multi-head
        batch_shape = q.shape[:-2]
        n, d = q.shape[-2], q.shape[-1]

        # flatten batch and head dimension
        q = einx.rearrange("... n d -> (...) n d", q)  # [num_heads * batch, n, d]
        k = einx.rearrange("... n d -> (...) n d", k)
        v = einx.rearrange("... n d -> (...) n d", v)

        batch_size = q.shape[0]  # num_heads * batch
        tq = (n + bq - 1) // bq
        tk = (n + bk - 1) // bk

        o = torch.empty((batch_size, n, d), device=q.device)
        l = torch.empty((batch_size, n), device=q.device)

        for i in range(tq):
            start_q, end_q = i * bq, min((i + 1) * bq, n)
            q_i = q[:, start_q:end_q]  # [batch_size, bq, d]

            o_i = torch.zeros((batch_size, end_q - start_q, d), device=q.device)
            l_i = torch.zeros((batch_size, end_q - start_q), device=q.device)
            m_i = torch.full(
                (batch_size, end_q - start_q), float("-inf"), device=q.device
            )

            for j in range(tk):
                start_k, end_k = j * bk, min((j + 1) * bk, n)
                k_j = k[:, start_k:end_k]  # [batch_size, bk, d]
                v_j = v[:, start_k:end_k]  # [batch_size, bk, d]

                s_ij = einx.dot("b bq d, b bk d -> b bq bk", q_i, k_j) / math.sqrt(
                    d
                )  # [batch_size, bq, bk]

                m_ij_max = s_ij.max(dim=-1).values  # [batch_size, bq]
                m_i_new = torch.maximum(m_i, m_ij_max)

                p = torch.exp(s_ij - m_i_new.unsqueeze(-1))  # [batch_size, bq, bk]

                scale = torch.exp(m_i - m_i_new)  # [batch_size, bq]
                l_i_new = scale * l_i + p.sum(dim=-1)  # [batch_size, bq]

                o_i = scale.unsqueeze(-1) * o_i + p @ v_j  # [batch_size, bq, d]

                l_i = l_i_new
                m_i = m_i_new

            o_i_normalized = o_i / (l_i.unsqueeze(-1) + eps)
            o[:, start_q:end_q] = o_i_normalized
            l[:, start_q:end_q] = torch.log(l_i + eps) + m_i

        o = o.view(*batch_shape, n, d)
        l = l.view(*batch_shape, n)

        ctx.save_for_backward(
            q.view(*batch_shape, n, d),
            k.view(*batch_shape, n, d),
            v.view(*batch_shape, n, d),
            l,
            o,
        )
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
