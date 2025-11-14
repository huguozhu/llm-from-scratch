from transformers import AutoTokenizer
import os

# 替换为你的实际路径和原始模型名
checkpoint_path = "checkpoints/math_sft"          # 你的训练输出目录
original_model_id = "Qwen/Qwen2-7B"               # 或你最初用的 model_id，比如 args.model

# 确保目录存在
os.makedirs(checkpoint_path, exist_ok=True)

# 从原始模型加载 tokenizer（必须加 trust_remote_code=True 如果是 Qwen）
tokenizer = AutoTokenizer.from_pretrained(
    original_model_id,
    trust_remote_code=True,
    use_fast=True  # 推荐，生成 tokenizer.json
)

# 保存到你的 checkpoint 目录
tokenizer.save_pretrained(checkpoint_path)

print(f"✅ Tokenizer saved to {checkpoint_path}")