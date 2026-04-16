# ==============================================================================
# 文本生成模块（推理/Inference）
# ==============================================================================
# 功能概述：
#   实现基于训练好的 Transformer 模型的自回归文本生成（Auto-Regressive Generation）。
#   给定一个提示文本（prompt），模型逐 token 地预测下一个词，直到遇到终止符
#   或达到最大序列长度。
#
# 生成策略：
#   采用 "Temperature Scaling + Top-p (Nucleus) Sampling" 组合策略：
#
#   1. Temperature Scaling（温度缩放）：
#      将 logits 除以温度系数 T，控制概率分布的平滑度：
#        - T < 1 : 分布更尖锐，倾向于选择高概率 token（更确定性）
#        - T = 1 : 原始分布
#        - T > 1 : 分布更平坦，增加随机性和多样性
#
#   2. Top-p (Nucleus) Sampling（核采样）：
#      按概率从高到低排序，只保留累积概率不超过 p 的 token 集合，
#      然后在这个缩小的候选集中重新归一化并采样。
#      相比 Top-k 采样，Top-p 的候选集大小是动态的，更灵活。
#
# 生成流程：
#   1. 加载训练好的模型检查点和 BPE 分词器
#   2. 将 prompt 编码为 token ID 序列
#   3. 循环生成：
#      a. 截取最后 max_seq_len 个 token 作为输入（滑动窗口）
#      b. 模型前向传播得到 logits
#      c. 取最后一个位置的 logits -> 温度缩放 -> Softmax -> Top-p 过滤 -> 采样
#      d. 将采样结果追加到序列末尾
#      e. 如果生成了 <|endoftext|> 则停止
#   4. 将生成的 token ID 解码为文本
# ==============================================================================
from torch.optim import AdamW
from llm.args import get_parser
from llm.checkpoint import load_checkpoint
from llm.transformer import Transformer, Softmax
from llm.bpe_tokenizer import BpeTokenizer
import torch
import os


def generate(prompt: str) -> tuple[str, list[int]]:
    """
    根据提示文本生成续写内容。

    参数：
        prompt : 用户提供的提示文本

    返回：
        (generated_text, generated_token_ids) : 生成的文本和对应的 token ID 列表

    生成参数由命令行参数控制：
        --temperature : 温度系数（默认 0.8）
        --top_p       : Top-p 采样阈值（默认 0.9）
        --max_seq_len : 最大生成长度（默认 512）
    """
    parser = get_parser()
    args = parser.parse_args()

    # 1. 构建模型并加载检查点
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        device=args.device,
    ).to(args.device)

    load_checkpoint(os.path.join(args.checkpoint_path, f"chpt_{str(args.iterations)}.pt"), model)

    tokenizer = BpeTokenizer()
    tokenizer.load(args.tokenizer_checkpoint)

    # Encode the prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(args.max_seq_len):
            # Get the last context_length tokens
            input_ids_cond = input_ids[:, -model.max_seq_len :]

            # The positions should be relative to the current context window
            token_positions = torch.arange(input_ids_cond.shape[1], device=args.device).unsqueeze(0)

            # Get the logits from the model
            logits = model(input_ids_cond, token_positions)
            # Take the logits for the last token
            logits = logits[:, -1, :]
            # print(logits)

            # Apply temperature scaling
            logits = logits / args.temperature

            # Apply top-p sampling
            probs = Softmax()(logits)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > args.top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[:, indices_to_remove] = 0

            # Re-normalize the probabilities
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for end-of-text token
            if next_token.item() == tokenizer.vcab2id[tokenizer.special_tokens[0]]:
                break

    # Decode the generated tokens
    prompt_len = len(token_ids)
    generated_ids = input_ids[0, prompt_len:].tolist()
    if generated_ids and generated_ids[-1] == tokenizer.vcab2id[tokenizer.special_tokens[0]]:
        generated_ids.pop()
    return tokenizer.decode(generated_ids), generated_ids


if __name__ == "__main__":
    prompt = "tell you a story"
    print(f"Prompt: {prompt}")
    output, output_token_ids = generate(prompt)
    print(f"Completion: {output}")
    print(output_token_ids)

