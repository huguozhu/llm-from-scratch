# ==============================================================================
# Transformer 模型核心模块
# ==============================================================================
# 功能概述：
#   从零实现一个 GPT 风格的 Decoder-Only Transformer 语言模型，包含以下组件：
#
#   【基础层】
#   - Linear       : 全连接线性层，Xavier 截断正态初始化，使用 einx 做矩阵乘法
#   - Embedding    : 词嵌入层，将 token ID 映射为 d_model 维向量
#   - RmsNorm      : RMS 归一化（Root Mean Square LayerNorm），LLaMA 风格，无均值偏移
#   - Softmax      : 数值稳定的 Softmax（减去最大值防止溢出）
#
#   【激活函数与前馈网络】
#   - SiLU         : Sigmoid Linear Unit，即 x * sigmoid(x)，也称 Swish
#   - Glu          : 门控线性单元，输出 = sigmoid(W1·x) * W2·x
#   - SwiGlu       : SwiGLU 前馈网络（LLaMA/Llama2 使用），输出 = W2·(SiLU(W1·x) * W3·x)
#                    包含三个线性变换：W1(门控), W3(值), W2(投影回 d_model)
#   - FFN          : SwiGlu 的别名，作为 TransformerBlock 中的前馈网络
#
#   【位置编码】
#   - RoPE         : 旋转位置编码（Rotary Position Embedding），Su et al. 2021
#                    通过将 Q/K 向量在二维子空间中旋转来注入位置信息
#                    支持变长序列（通过 token_positions 参数指定位置索引）
#                    预计算并缓存 cos/sin 表，推理时仅做查表和逐元素运算
#
#   【注意力机制】
#   - ScaledDotProductAttention : 缩放点积注意力
#       公式: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_k)) · V
#       支持因果掩码（causal mask），防止 token 看到未来位置的信息
#   - MultiHeadAttention        : 多头注意力，将 d_model 拆分为 num_head 个子空间并行计算
#       使用单次线性投影生成 Q/K/V（3*d_model），然后拆分为多头
#   - MultiHeadAttentionWithRoPE: 在 MultiHeadAttention 基础上，对 Q 和 K 施加 RoPE 旋转
#
#   【Transformer 结构】
#   - TransformerBlock : 单个 Transformer 解码器层
#       结构: x -> RmsNorm -> MHA(RoPE) -> 残差连接 -> RmsNorm -> SwiGLU FFN -> 残差连接
#       采用 Pre-Norm 架构（先归一化再做子层计算），训练更稳定
#   - Transformer      : 完整的 Decoder-Only Transformer 模型
#       结构: Embedding -> N × TransformerBlock -> RmsNorm -> Linear(vocab_size)
#       输入 token ID 序列，输出每个位置上的词表 logits
#
#   【损失函数】
#   - CrossEntropyLoss : 交叉熵损失，使用 log_softmax + 负对数似然
#       将 logits 展平为 (batch*seq, vocab)，targets 展平为 (batch*seq,)
#       计算所有 token 位置的平均负对数似然损失
#
#   【优化器】
#   - SGDDecay : 带学习率衰减的 SGD，lr 随步数 t 按 1/sqrt(t+1) 衰减
#   - AdamW    : 从零实现的 AdamW 优化器（解耦权重衰减版 Adam）
#       包含一阶动量 m (beta1 指数移动平均) 和二阶动量 v (beta2 指数移动平均)
#       偏差校正: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
#       参数更新: theta -= lr * m_hat / (sqrt(v_hat) + eps)
#       权重衰减: theta -= lr * weight_decay * theta （与梯度无关的解耦衰减）
#
#   【工具函数】
#   - cos_lr_scheduler : 余弦退火学习率调度器（带线性预热）
#       阶段1 (0 ~ warmup_iters): 线性预热，从 0 增长到 lr_max
#       阶段2 (warmup_iters ~ cos_cycle_iters): 余弦退火，从 lr_max 衰减到 lr_min
#       阶段3 (> cos_cycle_iters): 保持 lr_min 不变
#   - gradient_clip    : 梯度裁剪，当所有参数梯度的全局 L2 范数超过 max_norm 时，
#       按比例缩小所有梯度，防止梯度爆炸
# ==============================================================================
import torch
import math
import einx
from typing import overload, TypeAlias, Any
from collections.abc import Callable, Iterable
from torch import Tensor
from jaxtyping import Float

# 参数类型别名：支持传入张量列表、参数组字典列表、或 (name, tensor) 元组列表
ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]]


class Linear(torch.nn.Module):
    """
    全连接线性层（无偏置项）。
    权重初始化：使用 Xavier 截断正态分布 (Glorot)，sigma = sqrt(2/(in+out))，
    截断范围 [-3*sigma, 3*sigma]，避免极端初始值导致训练不稳定。
    前向计算：使用 einx.dot 执行矩阵乘法 x @ W^T，输出维度为 out_features。
    """
    def __init__(
        self, in_features, out_features, weights: Float[Tensor, " out in"] | None = None, device=None, dtype=None
    ):
        super().__init__()
        if weights is None:
            # Xavier 初始化标准差：保证前向/反向传播中方差不变
            sigma = math.sqrt(2.0 / (in_features + out_features))
            self.w = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            # 截断正态分布初始化，截断在 ±3sigma 处
            torch.nn.init.trunc_normal_(self.w, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        else:
            self.w = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # einx.dot: "... [in], out [in] -> ... out" 即 x @ W^T
        return einx.dot("... [in], out [in] -> ... out", x, self.w)


class Embedding(torch.nn.Module):
    """
    词嵌入层：将离散的 token ID 映射为连续的 d_model 维稠密向量。
    嵌入矩阵形状为 (vocab_size, d_model)，使用截断正态分布初始化，
    标准差 = 1/sqrt(d_model)，保证初始嵌入向量的模长适中。
    前向计算：通过索引操作 embeddings[token_ids] 实现查表。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1 / math.sqrt(embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 直接通过索引查表获取嵌入向量，等价于 one-hot @ embedding_matrix
        return self.embeddings[token_ids]


class RmsNorm(torch.nn.Module):
    """
    RMS 归一化（Root Mean Square Layer Normalization）。
    与标准 LayerNorm 的区别：不减去均值，仅除以均方根（RMS），计算更高效。
    公式: RmsNorm(x) = g * x / sqrt(mean(x^2) + eps)
    其中 g 是可学习的缩放参数（per-dimension），初始化为全 1。
    注意：计算时先转为 float32 避免半精度溢出，输出转回原始 dtype。
    """
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        # 可学习的逐维度缩放因子，初始化为 1
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)  # 提升精度避免 fp16/bf16 下溢
        variance = x.pow(2).mean(-1, keepdim=True)  # 计算每个位置的均方值
        x = x * torch.rsqrt(variance + self.eps)     # 除以 RMS（即 sqrt(variance)）
        return (self.g * x).to(input_dtype)


class SiLu(torch.nn.Module):
    """
    SiLU 激活函数（Sigmoid Linear Unit），也称 Swish。
    公式: SiLU(x) = x * sigmoid(x)
    特点：平滑、非单调、自门控（self-gated），在现代 LLM 中广泛使用。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


class Glu(torch.nn.Module):
    """
    门控线性单元（Gated Linear Unit, GLU）。
    公式: GLU(x) = sigmoid(W1·x) * W2·x
    其中 W1 产生门控信号（0~1），W2 产生候选值，两者逐元素相乘。
    相比普通 FFN，GLU 通过门控机制选择性地传递信息。
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features, out_features, device=device, dtype=dtype)  # 门控分支
        self.w2 = Linear(in_features, out_features, device=device, dtype=dtype)  # 值分支

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w1(x)) * self.w2(x)


class SwiGlu(torch.nn.Module):
    """
    SwiGLU 前馈网络（LLaMA / Llama2 / Llama3 使用的 FFN 变体）。
    公式: SwiGLU(x) = W2 · (SiLU(W1·x) * W3·x)

    结构说明：
      - W1 (d_in -> d_hidden) : 门控分支，经过 SiLU 激活
      - W3 (d_in -> d_hidden) : 值分支，不经过激活函数
      - W2 (d_hidden -> d_out): 投影回原始维度

    相比标准 FFN (Linear -> ReLU -> Linear)：
      - SwiGLU 多一个线性变换 W3，但中间维度可以适当减小来保持参数量相当
      - SiLU 替代 ReLU 避免了"死神经元"问题
      - 门控机制让网络能学习选择性激活
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_in, d_hidden, device=device, dtype=dtype)   # 门控分支
        self.w3 = Linear(d_in, d_hidden, device=device, dtype=dtype)   # 值分支
        self.w2 = Linear(d_hidden, d_out, device=device, dtype=dtype)  # 输出投影
        self.silu = SiLu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(W1·x) 作为门控信号，与 W3·x 逐元素相乘后投影回 d_out
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


# FFN 是 SwiGlu 的别名，在 TransformerBlock 中作为前馈网络使用
FFN = SwiGlu


class RoPE(torch.nn.Module):
    """
    旋转位置编码（Rotary Position Embedding, RoPE）- Su et al. 2021。

    核心思想：
      不像绝对位置编码那样直接加到 embedding 上，RoPE 通过在二维子空间中对 Q/K 向量
      进行旋转来注入位置信息。两个位置 m, n 的内积自然地只依赖于相对距离 m-n。

    数学原理：
      对于维度 d_k 的向量，将其看作 d_k/2 个二维子空间 (x_{2i}, x_{2i+1})。
      每个子空间旋转角度 theta_i(pos) = pos * theta_base^(-2i/d_k)。
      旋转矩阵：R(theta) = [[cos, -sin], [sin, cos]]

      应用方式：
        x_rot = x * cos(m*theta) + rotate_half(x) * sin(m*theta)
        其中 rotate_half 将 (a, b) 变为 (-b, a)

    实现细节：
      - 预计算 cos/sin 缓存表 (max_seq_len, dim)，注册为 buffer（不参与训练）
      - theta 默认 10000（与原始 Transformer 一致），可通过参数调整
      - 支持 token_positions 参数指定任意位置索引（推理时可能不连续）
      - 使用 repeat_interleave 将 (dim//2,) 的频率复制为 (dim,)，与 Q/K 维度匹配
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 逆频率向量: theta_i = 1 / (theta_base ^ (2i/d)), 形状 (dim//2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

        # 位置索引: [0, 1, 2, ..., max_seq_len-1], 形状 (seq_len,)
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 外积得到每个位置、每个频率的旋转角度, 形状 (seq_len, dim//2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 复制使得每个二维对共享同一频率, 形状 (seq_len, dim)
        emb = freqs.repeat_interleave(2, dim=-1)

        # 预计算 cos/sin 缓存，注册为 buffer（随模型移动到对应设备，但不参与梯度计算）
        self.register_buffer("cos_cached", emb.cos().to(dtype))
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Float[Tensor, "... seq"]) -> torch.Tensor:
        """
        对输入的 Q 或 K 向量施加旋转位置编码。

        参数：
            x               : (..., seq_len, dim) 的 Q 或 K 张量
            token_positions : (..., seq_len) 每个 token 的位置索引

        返回：
            旋转后的张量，形状与 x 相同
        """
        # 根据位置索引查表获取对应的 cos/sin 值
        cos = self.cos_cached[token_positions]  # (..., seq_len, dim)
        sin = self.sin_cached[token_positions]

        # 将最后一维拆成 (dim//2, 2) 的二维对
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        # 旋转: (a, b) -> (-b, a)，即逆时针旋转 90 度
        x_rotated = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)
        x_rotated = x_rotated.view(*x.shape)

        # 处理多头注意力的维度广播 (batch, num_heads, seq, dim)
        if x.ndim == 4:
            cos = cos.unsqueeze(1)  # 在 num_heads 维度上广播
            sin = sin.unsqueeze(1)

        # 旋转公式: x * cos(theta) + rotate_half(x) * sin(theta)
        x_rot = x * cos + x_rotated * sin
        return x_rot


class Softmax(torch.nn.Module):
    """
    数值稳定的 Softmax 实现。
    先减去最大值防止 exp 溢出（数值稳定性技巧），再做标准的指数归一化。
    公式: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        max_x = torch.max(x, dim, keepdim=True).values  # 数值稳定：减去最大值
        x = x - max_x
        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        return x_exp / x_exp_sum


class ScaledDotProductAttention(torch.nn.Module):
    """
    缩放点积注意力机制（Scaled Dot-Product Attention）。

    公式: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    工作流程：
      1. 计算 Q 和 K 的点积相似度矩阵: (seq_q, seq_k)
      2. 除以 sqrt(d_k) 缩放，防止维度过大时点积值过大导致 softmax 梯度消失
      3. 施加因果掩码（mask=True 的位置设为 -1e9，softmax 后趋近于 0）
      4. 对 softmax 后的注意力权重与 V 做加权求和

    参数：
        mask: bool 张量，True 表示该位置被遮蔽（不参与 softmax 计算）。
              对于因果（自回归）模型，上三角为 True，防止位置 i 看到 j > i 的信息。
    """
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        q: Float[Tensor, "... s d"],
        k: Float[Tensor, "... s d"],
        v: Float[Tensor, "... s d"],
        mask: torch.Tensor
        | None = None,
    ) -> torch.Tensor:
        d_model = q.shape[-1]

        # 步骤 1: Q·K^T 计算注意力分数矩阵，形状 (..., seq_q, seq_k)
        att = einx.dot("... s_q [d], ... s_k [d] -> ... s_q s_k", q, k)
        # 步骤 2: 缩放，除以 sqrt(d_k)
        att_scale = att / math.sqrt(d_model)

        # 步骤 3: 施加掩码（因果掩码：上三角设为极小值，softmax 后接近 0）
        if mask is not None:
            if mask.ndim < att_scale.ndim:
                mask = mask.reshape((1,) * (att_scale.ndim - mask.ndim) + mask.shape)
            att_scale = att_scale.masked_fill(mask, -1e9)

        # 步骤 4: Softmax 归一化得到注意力权重
        att_score = self.softmax(att_scale)

        # 步骤 5: 注意力权重 @ V，得到加权聚合后的输出
        return einx.dot("... s_q [s], ... [s] d -> ... s_q d", att_score, v)


class MultiHeadAttention(torch.nn.Module):
    """
    多头注意力机制（Multi-Head Attention, MHA）。

    核心思想：将 d_model 维空间拆成 num_head 个 d_k 维子空间（d_k = d_model/num_head），
    每个头独立做缩放点积注意力，最后拼接并投影回 d_model 维。

    实现特点：
      - 使用单个 Linear(d_model, 3*d_model) 一次性投影得到 Q/K/V，减少矩阵乘法次数
      - 使用 einx.rearrange 将 (batch, seq, 3*d_model) 拆分为 3 个 (batch, num_head, seq, d_k)
      - 因果掩码（上三角矩阵）预计算并注册为 buffer，避免重复创建
    """
    def __init__(self, d_model: int, num_head: int, max_seq_len=2048, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)   # 输出投影 W_O
        self.project = Linear(in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype)  # QKV 联合投影
        self.dot_product_att = ScaledDotProductAttention()

        # 预计算因果掩码：上三角为 True（被遮蔽），对角线及以下为 False（可见）
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Float[Tensor, "b s d"]) -> torch.Tensor:
        seq_len = x.shape[1]

        # 截取当前序列长度对应的因果掩码
        mask = self.causal_mask[:seq_len, :seq_len]

        # 一次线性投影得到 Q/K/V 的拼接，然后拆分为 3 个多头张量
        qkv = self.project(x)  # (batch, seq, 3*d_model)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        # 每个头独立做缩放点积注意力
        output = self.dot_product_att(q, k, v, mask)  # (batch, num_head, seq, d_k)
        # 拼接所有头的输出
        output = einx.rearrange("b h s d -> b s (h d)", output)  # (batch, seq, d_model)
        # 输出投影
        return self.out_linear(output)


class MultiHeadAttentionWithRoPE(MultiHeadAttention):
    """
    带有旋转位置编码（RoPE）的多头注意力。

    继承 MultiHeadAttention，在计算注意力之前对 Q 和 K 施加 RoPE 旋转。
    V 不施加 RoPE，因为位置信息只需要在计算注意力分数时起作用。

    RoPE 的维度 = d_model // num_head（即每个头的维度 d_k），
    这样每个头独立旋转自己的 Q/K 子空间。
    """
    def __init__(self, d_model: int, num_head: int, theta: float = 10000, max_seq_len=2048, device=None, dtype=None):
        super().__init__(d_model=d_model, num_head=num_head, max_seq_len=max_seq_len, device=device, dtype=dtype)
        # RoPE 作用在每个头的 d_k 维度上
        self.rope = RoPE(d_model // num_head, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        # 如果未提供位置索引，默认为 [0, 1, 2, ..., seq_len-1]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        # 仅对 Q 和 K 施加 RoPE 旋转，V 保持不变
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        output = self.dot_product_att(q, k, v, mask)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        return self.out_linear(output)


class TransformerBlock(torch.nn.Module):
    """
    单个 Transformer 解码器层（Decoder Block）。

    采用 Pre-Norm 架构（LLaMA 风格），计算流程：
      x -> RmsNorm -> Multi-Head Attention (RoPE) -> 残差连接
        -> RmsNorm -> SwiGLU FFN -> 残差连接

    Pre-Norm vs Post-Norm：
      - Pre-Norm（先归一化再计算子层）训练更稳定，不易出现梯度爆炸
      - Post-Norm（先计算子层再归一化）是原始 Transformer 的方式
      - 现代 LLM（GPT-3, LLaMA, Llama2 等）普遍使用 Pre-Norm

    残差连接：保留原始信息流，缓解深层网络的梯度消失问题
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rms_norm1 = RmsNorm(d_model, device=device, dtype=dtype)  # 注意力前的归一化
        self.rms_norm2 = RmsNorm(d_model, device=device, dtype=dtype)  # FFN 前的归一化
        self.mult_head_atten = MultiHeadAttentionWithRoPE(
            d_model, num_heads, theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ffe = FFN(d_model, d_ff, d_model, device=device, dtype=dtype)  # SwiGLU 前馈网络

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # 子层 1: 多头自注意力 + 残差连接
        x_norm = self.rms_norm1(x)
        x_atten = self.mult_head_atten(x_norm, token_positions)
        x = x + x_atten  # 残差连接

        # 子层 2: SwiGLU 前馈网络 + 残差连接
        x_norm = self.rms_norm2(x)
        x_ffe = self.ffe(x_norm)
        return x + x_ffe  # 残差连接


class Transformer(torch.nn.Module):
    """
    完整的 Decoder-Only Transformer 语言模型。

    整体架构（LLaMA 风格）：
      token_ids -> Embedding -> [TransformerBlock × N] -> RmsNorm -> Linear -> logits

    各组件说明：
      - Embedding  : 将 token ID 映射为 d_model 维向量（不使用额外的位置编码，RoPE 在注意力中施加）
      - blocks     : N 个 TransformerBlock 堆叠，N = num_layers
      - norm       : 最后的 RmsNorm，在投影到词表之前归一化
      - out_linear : 线性投影层 (d_model -> vocab_size)，输出每个位置对词表的 logits

    输入: token_ids (batch, seq_len) - 整数张量
    输出: logits (batch, seq_len, vocab_size) - 每个位置对所有词的未归一化对数概率

    模型规模（默认参数）：
      d_model=288, num_heads=6, d_ff=1024, num_layers=6, vocab_size=2048
      每个注意力头维度 d_k = 288/6 = 48
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        num_layers: int,
        max_seq_len=2048,
        rope_theta: float = 10000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 词嵌入层：vocab_size -> d_model
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        # N 个 Transformer 解码器层
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # 最终归一化层
        self.norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        # 输出投影层：d_model -> vocab_size
        self.out_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播：输入 token ID 序列，输出每个位置对词表的 logits。

        参数：
            token_ids      : (batch, seq_len) 整数张量
            token_positions: (batch, seq_len) 可选的位置索引（推理时可能不连续）

        返回：
            logits : (batch, seq_len, vocab_size) 未归一化的对数概率
        """
        # 词嵌入查表
        x = self.embedding(token_ids)
        # 默认位置索引 [0, 1, 2, ..., seq_len-1]
        if token_positions is None:
            batch_size, seq_len = token_ids.shape
            token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        # 逐层通过 Transformer 解码器
        for block in self.blocks:
            x = block(x, token_positions)
        # 最终归一化 + 投影到词表
        x_norm = self.norm(x)
        logits = self.out_linear(x_norm)
        return logits


class CrossEntropyLoss(torch.nn.Module):
    """
    交叉熵损失函数（从零实现）。

    计算流程：
      1. 将 logits 展平为 (batch*seq, vocab_size)，targets 展平为 (batch*seq,)
      2. 对 logits 做 log_softmax 得到对数概率分布
      3. 按 targets 索引取出每个位置正确类别的对数概率
      4. 取负号（负对数似然 NLL）并求均值

    数学公式：
      L = -1/N * sum_i log(softmax(logits_i)[target_i])

    等价于 PyTorch 的 nn.CrossEntropyLoss，但手动实现便于理解和调试。
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 展平: (batch, seq, vocab) -> (batch*seq, vocab)
        logits = einx.rearrange("... c -> (...) c", logits)
        # 展平: (batch, seq) -> (batch*seq,)
        targets = einx.rearrange("... -> (...)", targets)

        # log_softmax: 数值稳定的对数概率计算
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # 取出每个样本对应正确类别的对数概率
        correct_log_probs = log_probs[torch.arange(len(log_probs)), targets]
        # 负对数似然
        nll = -correct_log_probs

        # 所有 token 位置的平均损失
        mean_loss = torch.mean(nll)

        return mean_loss


class SGDDecay(torch.optim.Optimizer):
    """
    带学习率衰减的随机梯度下降（SGD with Decay）。
    学习率随步数 t 按 lr / sqrt(t+1) 衰减，使得早期快速学习，后期精细调整。
    主要用于实验/调试目的，实际训练使用 AdamW。
    """
    def __init__(self, params, lr=1e-3) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    """
    从零实现的 AdamW 优化器（解耦权重衰减版 Adam）。

    AdamW 与 Adam 的区别：
      - Adam  : 权重衰减嵌入梯度更新，即 grad += weight_decay * param
      - AdamW : 权重衰减独立于梯度更新，即 param -= lr * weight_decay * param
      AdamW 的解耦方式在使用自适应学习率时效果更好（Loshchilov & Hutter, 2019）

    核心算法（每步更新）：
      1. m = beta1 * m + (1-beta1) * grad          # 一阶动量（梯度的指数移动平均）
      2. v = beta2 * v + (1-beta2) * grad^2         # 二阶动量（梯度平方的指数移动平均）
      3. m_hat = m / (1 - beta1^t)                  # 偏差校正（消除初始零偏置）
      4. v_hat = v / (1 - beta2^t)                  # 偏差校正
      5. param -= lr * m_hat / (sqrt(v_hat) + eps)  # 自适应学习率更新
      6. param -= lr * weight_decay * param          # 解耦权重衰减

    参数：
      lr           : 学习率（默认 1e-3）
      betas        : (beta1, beta2) 动量衰减系数（默认 (0.9, 0.999)）
      weight_decay : 权重衰减系数（默认 1e-3）
      eps          : 数值稳定常数（默认 1e-8），防止除以零
    """
    def __init__(
        self,
        params: ParamsT,
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["sm"] = torch.zeros_like(p.data)

                m, sm = state["m"], state["sm"]
                t = state["t"] + 1

                grad = p.grad.data

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Update biased second raw moment estimate
                sm.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1**t)
                sm_hat = sm / (1.0 - beta2**t)

                # Update parameters
                p.data.addcdiv_(m_hat, torch.sqrt(sm_hat) + eps, value=-lr)

                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                state["t"] = t
        return loss


def cos_lr_scheduler(it: int, warmup_iters: int, cos_cycle_iters: int, lr_min: float, lr_max: float) -> float:
    """
    余弦退火学习率调度器（带线性预热）。

    三个阶段：
      阶段 1 (0 ~ warmup_iters):
        线性预热，lr 从 0 线性增长到 lr_max
        公式: lr = lr_max * it / warmup_iters

      阶段 2 (warmup_iters ~ cos_cycle_iters):
        余弦退火，lr 从 lr_max 平滑下降到 lr_min
        公式: lr = lr_min + 0.5*(lr_max-lr_min)*(1 + cos(pi * progress))
        其中 progress = (it - warmup_iters) / (cos_cycle_iters - warmup_iters)

      阶段 3 (> cos_cycle_iters):
        保持最小学习率 lr_min 不变

    参数：
        it              : 当前迭代步数
        warmup_iters    : 预热阶段总步数
        cos_cycle_iters : 余弦周期总步数（含预热）
        lr_min          : 最小学习率（退火终值）
        lr_max          : 最大学习率（预热目标值）

    返回：
        当前步数对应的学习率
    """
    if it <= warmup_iters:
        # 阶段 1: 线性预热
        return lr_max * it / warmup_iters
    elif warmup_iters < it < cos_cycle_iters:
        # 阶段 2: 余弦退火
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cos_cycle_iters - warmup_iters))
        )
    else:
        # 阶段 3: 保持最小学习率
        return lr_min


def gradient_clip(params: Iterable[torch.nn.Parameter], max_norm: float, delta=1e-6):
    """
    全局梯度裁剪（Gradient Clipping by Global Norm）。

    计算所有参数梯度的全局 L2 范数，如果超过 max_norm 则按比例缩小所有梯度。
    目的：防止梯度爆炸，稳定训练过程。

    算法：
      1. total_norm = sqrt(sum(||grad_i||^2))  （所有参数梯度的全局范数）
      2. 如果 total_norm > max_norm:
           clip_coef = max_norm / total_norm
           每个梯度 *= clip_coef
         否则不做裁剪

    参数：
        params   : 模型参数迭代器
        max_norm : 梯度全局范数的最大阈值（默认 1.0）
        delta    : 数值稳定常数，防止除以零
    """
    with torch.no_grad():
        grads = [p.grad for p in params if p.grad is not None]
        # 计算全局梯度 L2 范数
        total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach()) for g in grads]))
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + delta)
            for g in grads:
                g.detach().mul_(clip_coef)  # 按比例缩小每个梯度


if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGDDecay([weights], lr=lr)
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
            print(f"lr={lr}, t={t}, loss={loss}")
