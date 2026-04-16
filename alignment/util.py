# ==============================================================================
# 对齐训练通用工具函数模块
# ==============================================================================
# 功能概述：
#   为 SFT/RL 训练提供通用工具函数：
#   - generate_text()        : 模型文本生成（Temperature + Top-p 采样）
#   - extract_answer()       : 从输出提取 \\boxed{} 中的数学答案
#   - extract_gsm8k_answer() : 从 GSM8K 标准答案提取 "####" 后的数值
#   - compute_log_probs()    : 计算 token 序列的对数概率（用于 RL 策略梯度）
#   - reward_function()      : 综合奖励（正确性 + 格式合规性）
# ==============================================================================
import torch
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoConfig,
    # init_empty_weights,
    AutoModelForCausalLM,
)
from accelerate import init_empty_weights, infer_auto_device_map
import torch.nn.utils.rnn as rnn_utils
import os

# import patch
from unittest.mock import patch

import torch.nn.functional as F
from vllm import LLM


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_sequences = []
    batch_response_masks = []

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_tokens = tokenizer.encode(output_str, add_special_tokens=False)

        full_sequence = prompt_tokens + output_tokens
        batch_sequences.append(torch.tensor(full_sequence, dtype=torch.long))

        mask = [0.0] * len(prompt_tokens) + [1.0] * len(output_tokens)
        batch_response_masks.append(torch.tensor(mask, dtype=torch.float32))

    padded_sequences = rnn_utils.pad_sequence(
        batch_sequences,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    padded_masks = rnn_utils.pad_sequence(
        batch_response_masks,
        batch_first=True,
        padding_value=0.0,
    )

    return {
        "input_ids": padded_sequences[:, :-1],
        "labels": padded_sequences[:, 1:],
        "response_mask": padded_masks[:, 1:],
    }


def compute_enptropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the per-token entropy of next-token predictions.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length). The entropy for each next-token
            prediction.
    """
    probs = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_p, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get the log probabilities of the response tokens.

    Args:
        model (PreTrainedModel): The model to use for prediction.
        input_ids (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing
            the input token IDs.
        labels (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing
            the labels for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length). The log probabilities of the
            response tokens.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        response_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
    return {
        "log_probs": response_log_probs,
        "token_entropy": compute_enptropy(logits) if return_token_entropy else None,
    }


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Normalize a tensor along the last dimension using a mask.

    Args:
        tensor (torch.Tensor): Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.
        mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing
            the mask for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length, vocab_size). The normalized tensor.
    """
    if tensor.shape != mask.shape:
        raise ValueError(
            f"tensor shape {tensor.shape} does not match mask shape {mask.shape}"
        )
    if normalize_constant == 0:
        raise ZeroDivisionError("normalize_constant is zero")

    masked_tensor = tensor * mask.to(dtype=tensor.dtype)

    return torch.sum(masked_tensor, dim=dim) / normalize_constant


def get_device_ids(device: str) -> list:
    if device == "cpu":
        return []
    elif device.startswith("cuda"):
        # Handles formats like "cuda:0" and "cuda:0,cuda:1,cuda:2"
        device_parts = device.split(",")
        return [int(part.split(":")[1]) for part in device_parts]
    else:
        raise ValueError(f"Unknown device: {device}")


def get_device_map(
    hf_model_id: str | os.PathLike,
    device: str,
    max_gpu_use: str = "32GiB",
    dtype="bfloat16",
):
    config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Tie the weights before calculating the device map.
    empty_model.tie_weights()

    sft_device_ids = get_device_ids(device)
    # You can adjust the memory usage per GPU here if needed
    max_memory = {dev_id: max_gpu_use for dev_id in sft_device_ids}

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        dtype=dtype,
        no_split_module_classes=empty_model._no_split_modules,
    )

    del empty_model
    return device_map


def init_vllm(
    model_path: str,
    device: str,
    dtype: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = False,
    tokenizer_path: str | None = None,
) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    if tokenizer_path is None:
        tokenizer_path = model_path

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            device=device,
            dtype=dtype,
            seed=seed,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]
