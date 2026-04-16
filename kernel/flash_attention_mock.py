# ==============================================================================
# Flash Attention 纯 PyTorch 模拟实现
# ==============================================================================
# 功能概述：
#   用纯 PyTorch 实现 Flash Attention 算法（Dao et al., 2022），包含自定义前向和反向传播。
#   主要用于功能验证、正确性对比和无 Triton 环境下的回退方案。
#
# Flash Attention 核心思想：
#   标准 Attention 需要 O(N^2) 显存存储完整的注意力矩阵 S = QK^T，
#   当序列长度 N 很大时（如 4096+），这会成为显存瓶颈。
#   Flash Attention 通过分块（Tiling）+ 在线 Softmax（Online Softmax）技巧，
#   逐块计算注意力，全程不需要物化完整的 N×N 矩阵，显存降为 O(N)。
#
# 在线 Softmax 算法：
#   对每个 Q 块，遍历所有 K 块，增量更新三个统计量：
#     - m_i : 当前最大值（用于数值稳定性）
#     - l_i : softmax 分母的累积和（未归一化）
#     - o_i : 加权 V 的累积和（未归一化）
#   每当新块的最大值更大时，需要对之前累积的 l_i 和 o_i 做缩放校正：
#     scale = exp(m_old - m_new)
#     l_i_new = scale * l_i + sum(exp(s_ij - m_new))
#     o_i_new = scale * o_i + exp(s_ij - m_new) @ v_j
#   最后归一化：o_i / l_i
#
# 包含两种前向模式：
#   - forward_naive : 标准实现（物化完整注意力矩阵），用于正确性验证
#   - forward       : 分块 Flash Attention 实现（默认），支持自定义块大小 bq/bk
#
# 反向传播：
#   使用 @torch.compile 加速的标准注意力反向传播公式，
#   计算 dQ, dK, dV（支持因果掩码）。
#
# 保存的中间结果（用于反向传播）：
#   - q, k, v : 输入张量
#   - l       : log-sum-exp 值（数值稳定的 softmax 分母对数）
#   - o       : 前向输出
# ==============================================================================
import torch
from torch import Tensor
from jaxtyping import Float
import math
import einx


class FlashAttentionMock(torch.autograd.Function):
    """
    Flash Attention 的纯 PyTorch 实现（自定义 autograd Function）。
    支持多头注意力（通过将 batch 和 head 维度展平处理）。
    """
    @staticmethod
    def forward_naive(
        ctx,
        q: Float[Tensor, "... n d"],
        k: Float[Tensor, "... n d"],
        v: Float[Tensor, "... n d"],
        is_causal: bool,
    ):
        """
        朴素前向传播：物化完整的 N×N 注意力矩阵，用于正确性验证。
        显存复杂度 O(N^2)。
        """
        d = q.shape[-1]
        # 标准缩放点积注意力
        p = einx.dot("... n d, ... m d -> ... n m", q, k) / math.sqrt(d)
        o = torch.nn.functional.softmax(p, dim=-1) @ v
        # 保存 log-sum-exp 用于反向传播
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
        """
        分块 Flash Attention 前向传播。

        参数：
            q, k, v   : 查询/键/值张量 (..., seq_len, d_model)
            is_causal  : 是否使用因果掩码（暂未在分块版本中实现）
            naive      : 若为 True 则回退到朴素实现
            bq, bk     : Q 块大小和 K 块大小（默认各 16）
            eps        : 数值稳定常数

        算法流程（在线 Softmax + 分块累积）：
            对于每个 Q 块 i (共 tq = ceil(n/bq) 个)：
                初始化 m_i = -inf, l_i = 0, o_i = 0
                遍历每个 K 块 j (共 tk = ceil(n/bk) 个)：
                    s_ij = q_i @ k_j^T / sqrt(d)           # 局部注意力分数
                    m_new = max(m_i, max(s_ij))             # 更新最大值
                    scale = exp(m_old - m_new)              # 校正因子
                    p_ij = exp(s_ij - m_new)                # 局部 softmax 分子
                    l_new = scale * l_i + sum(p_ij)         # 更新分母
                    o_new = scale * o_i + p_ij @ v_j        # 更新加权和
                最终: o_i = o_i / l_i                       # 归一化
        """
        if naive:
            return FlashAttentionMock.forward_naive(ctx, q, k, v, is_causal)

        # 展平 batch 和 head 维度以统一处理
        batch_shape = q.shape[:-2]
        n, d = q.shape[-2], q.shape[-1]

        q = einx.rearrange("... n d -> (...) n d", q)
        k = einx.rearrange("... n d -> (...) n d", k)
        v = einx.rearrange("... n d -> (...) n d", v)

        batch_size = q.shape[0]
        tq = (n + bq - 1) // bq  # Q 块数
        tk = (n + bk - 1) // bk  # K 块数

        o = torch.empty((batch_size, n, d), device=q.device)
        l = torch.empty((batch_size, n), device=q.device)

        for i in range(tq):
            start_q, end_q = i * bq, min((i + 1) * bq, n)
            q_i = q[:, start_q:end_q]

            # 在线 Softmax 的三个累积量
            o_i = torch.zeros((batch_size, end_q - start_q, d), device=q.device)
            l_i = torch.zeros((batch_size, end_q - start_q), device=q.device)
            m_i = torch.full(
                (batch_size, end_q - start_q), float("-inf"), device=q.device
            )

            for j in range(tk):
                start_k, end_k = j * bk, min((j + 1) * bk, n)
                k_j = k[:, start_k:end_k]
                v_j = v[:, start_k:end_k]

                # 局部注意力分数
                s_ij = einx.dot("b bq d, b bk d -> b bq bk", q_i, k_j) / math.sqrt(d)

                # 在线 Softmax 更新
                m_ij_max = s_ij.max(dim=-1).values
                m_i_new = torch.maximum(m_i, m_ij_max)

                p = torch.exp(s_ij - m_i_new.unsqueeze(-1))  # 局部 softmax 分子

                scale = torch.exp(m_i - m_i_new)  # 校正旧累积量
                l_i_new = scale * l_i + p.sum(dim=-1)

                o_i = scale.unsqueeze(-1) * o_i + p @ v_j

                l_i = l_i_new
                m_i = m_i_new

            # 归一化并写入输出
            o_i_normalized = o_i / (l_i.unsqueeze(-1) + eps)
            o[:, start_q:end_q] = o_i_normalized
            l[:, start_q:end_q] = torch.log(l_i + eps) + m_i

        # 恢复原始 batch 形状
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
        """反向传播：计算 dQ, dK, dV，使用 torch.compile 加速。"""
        q, k, v, l, o = ctx.saved_tensors
        is_causal = ctx.is_causal
        b, n, d = q.shape
        scale = 1.0 / math.sqrt(d)

        dQ, dK, dV = _flash_attn_backward_compiled(
            q, k, v, o, grad_out[0], l, scale, is_causal
        )
        return dQ, dK, dV, None


@torch.compile
def _flash_attn_backward_compiled(q, k, v, o, do, l, scale, is_causal):
    """
    Flash Attention 反向传播（torch.compile 加速版）。
    通过重新计算注意力权重 P 来避免存储 O(N^2) 的中间矩阵。

    计算公式：
        dV = P^T @ dO
        dP = dO @ V^T
        dS = P * (dP - sum(P * dP, dim=-1))   # softmax 反向传播
        dQ = dS @ K * scale
        dK = dS^T @ Q * scale
    """
    b, n, d = q.shape

    # 重新计算注意力分数
    s = einx.dot("b q d, b k d -> b q k", q, k) * scale

    if is_causal:
        causal_mask = torch.ones_like(s, dtype=torch.bool).triu_(diagonal=1)
        s = s.masked_fill(causal_mask, float("-inf"))

    # 数值稳定的 softmax 重计算
    s_max = s.amax(dim=-1, keepdim=True).detach()
    p_unmasked = (s - s_max).exp()
    p_sum = p_unmasked.sum(dim=-1, keepdim=True)
    p = p_unmasked / (p_sum + 1e-10)

    if is_causal:
        p = p.masked_fill(causal_mask, 0.0)

    # dV = P^T @ dO
    dv = einx.dot("b q k, b q d -> b k d", p, do)
    # dP = dO @ V^T
    dp = einx.dot("b q d, b k d -> b q k", do, v)

    if is_causal:
        dp = dp.masked_fill(causal_mask, 0.0)

    # Softmax 反向传播: dS = P * (dP - row_sum(P * dP))
    p_dot_dp = torch.einsum("b q k, b q k -> b q", p, dp).unsqueeze(-1)
    ds = p * (dp - p_dot_dp)

    # dQ = dS @ K, dK = dS^T @ Q
    dq = einx.dot("b q k, b k d -> b q d", ds, k) * scale
    dk = einx.dot("b q k, b q d -> b k d", ds, q) * scale

    return dq, dk, dv
