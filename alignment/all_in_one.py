import argparse
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from vllm import SamplingParams, LLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

# We no longer need accelerate's inference utils
# from accelerate import init_empty_weights, infer_auto_device_map
import math  # Import for ceiling division

from alignment.args import get_rl_parser
from alignment.dataset import Gsm8kDataset
from alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from alignment.sft import init_vllm
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.evaluate import evaluate_math


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def train(args):
    dtype = str_to_torch_dtype(args.dtype)
    group_size = 1 if args.loss_type == "no_baseline" else args.group_size

    # === Step 1: Initialize vLLM for sampling (rollout) ===
    print(f"Initializing vLLM sample_model on {args.sample_device}...")
    sample_model = init_vllm(
        model_path=args.model,
        dtype=args.dtype,
        seed=args.seed,
        device=args.sample_device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    # === Step 2: Load trainable policy model with a MANUALLY BALANCED device map ===
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Robust Dynamic GPU Allocation ---
    total_gpu_count = torch.cuda.device_count()
    if total_gpu_count < 3:
        raise ValueError(
            "This script requires at least 3 GPUs: 1 for sampling, 1 for reference, and at least 1 for the policy model."
        )

    try:
        sample_device_idx = int(args.sample_device.split(":")[-1])
        ref_device_idx = int(args.reference_model_device.split(":")[-1])
    except (ValueError, IndexError):
        raise ValueError(
            "Invalid device format. Please use 'cuda:N' for device arguments."
        )

    reserved_indices = {sample_device_idx, ref_device_idx}
    all_gpu_indices = set(range(total_gpu_count))
    policy_gpu_indices = sorted(list(all_gpu_indices - reserved_indices))

    if not policy_gpu_indices:
        raise ValueError(
            "No GPUs are left for the policy model after reserving for sampling and reference."
        )

    policy_devices_str = [f"cuda:{i}" for i in policy_gpu_indices]

    print("-" * 60)
    print(f"Total GPUs detected: {total_gpu_count}")
    print(f"Reserved for vLLM sampling: cuda:{sample_device_idx}")
    print(f"Reserved for reference model: cuda:{ref_device_idx}")
    print(f"MANUALLY balancing policy model across: {policy_devices_str}")
    print("-" * 60)

    ### MODIFICATION: Build the device_map manually for perfect balance ###
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # 1. Designate a main GPU for embeddings and the final output.
    main_gpu = policy_gpu_indices[0]
    # The rest of the GPUs will handle the transformer layers.
    layer_gpus = policy_gpu_indices[1:]

    # If there's only one policy GPU, it does everything.
    if not layer_gpus:
        layer_gpus = [main_gpu]

    # 2. Calculate how many layers to put on each of the layer_gpus.
    num_layers = config.num_hidden_layers
    layers_per_gpu = math.ceil(num_layers / len(layer_gpus))

    device_map = {}

    # 3. Assign the memory-critical start and end layers to the main GPU.
    device_map["model.embed_tokens"] = main_gpu
    device_map["lm_head"] = main_gpu
    device_map["model.norm"] = main_gpu

    # 4. Distribute the transformer layers evenly across the other GPUs.
    gpu_idx_for_layers = 0
    for i in range(num_layers):
        # Move to the next GPU when the current one is full.
        if i > 0 and i % layers_per_gpu == 0:
            gpu_idx_for_layers += 1
        gpu_to_assign = layer_gpus[gpu_idx_for_layers]
        device_map[f"model.layers.{i}"] = gpu_to_assign

    print("Manually constructed balanced device map:")
    print(device_map)
    ### END MODIFICATION ###

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )

    model.train()
    policy_device = model.device

    # === Step 3: Load reference model (frozen) on its dedicated GPU ===
    print(f"Loading reference_model on {args.reference_model_device}...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": args.reference_model_device},
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # ... The rest of your training loop is correct and remains unchanged ...
    # ...
    # === Dataset & DataLoader ===
    train_dataset = Gsm8kDataset(
        data_path=args.rl_train_data,
        promt_template_path=args.prompt_template_path,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # === Optimizer ===
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    global_step = 0
    for epoch in range(args.epochs):
        epoch_pbar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_id, batch in enumerate(epoch_pbar):
            prompts, _, ground_truths = batch

            grouped_prompts = [p for p in prompts for _ in range(group_size)]
            grouped_ground_truths = [
                gt for gt in ground_truths for _ in range(group_size)
            ]

            # === Rollout with vLLM ===
            sampling_params = SamplingParams(
                temperature=args.sampling_temperature,
                min_tokens=args.sampling_min_tokens,
                max_tokens=args.sampling_max_tokens,
            )
            vllm_outputs = sample_model.generate(grouped_prompts, sampling_params)
            rollout_responses = [output.outputs[0].text for output in vllm_outputs]
            full_texts = [p + r for p, r in zip(grouped_prompts, rollout_responses)]

            # === Compute rewards and advantages ===
            advantages, raw_rewards, _ = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=grouped_ground_truths,
                group_size=group_size,
                advantage_eps=args.advantage_eps,
                normalize_by_std=args.use_std_normalization,
            )

            # === Tokenize ===
            prompt_tokens = tokenizer(
                grouped_prompts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            )
            prompt_lengths = prompt_tokens.attention_mask.sum(dim=1)

            full_tokens = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            )

            # Move to correct devices
            input_ids = full_tokens["input_ids"].to(policy_device)
            attention_mask = full_tokens["attention_mask"].to(policy_device)
            advantages = advantages.to(policy_device)
            raw_rewards = raw_rewards.to(policy_device)
            prompt_lengths = prompt_lengths.to(policy_device)

            # === Get logits from policy model ===
            policy_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # === Get old logits from reference model ===
            with torch.inference_mode():
                ref_input_ids = full_tokens["input_ids"].to(args.reference_model_device)
                ref_attention_mask = full_tokens["attention_mask"].to(
                    args.reference_model_device
                )
                old_logits = reference_model(
                    input_ids=ref_input_ids, attention_mask=ref_attention_mask
                ).logits
            old_logits = old_logits.to(
                policy_device
            )  # align with policy for logprob computation

            # === Compute log probs ===
            policy_log_probs_all = torch.log_softmax(policy_logits, dim=-1)
            old_log_probs_all = torch.log_softmax(old_logits, dim=-1)

            policy_log_probs = torch.gather(
                policy_log_probs_all, -1, input_ids.unsqueeze(-1)
            ).squeeze(-1)
            old_log_probs = torch.gather(
                old_log_probs_all, -1, input_ids.unsqueeze(-1)
            ).squeeze(-1)

            # === Response mask ===
            response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for i in range(len(response_mask)):
                start = prompt_lengths[i].item()
                response_mask[i, start:] = True
            response_mask &= input_ids != tokenizer.pad_token_id

            # print("Prompt lengths:", prompt_lengths[:5])
            # print("Response mask sum per sample:", response_mask.sum(dim=1)[:5])
            # assert response_mask.sum() > 0, "No response tokens found!"

            # === Micro-batch training ===
            total_samples = len(full_texts)
            total_loss = 0.0

            optimizer.zero_grad()

            for i in range(0, total_samples, args.train_mini_batch_size):
                end = min(i + args.train_mini_batch_size, total_samples)
                mb_policy_log_probs = policy_log_probs[i:end]
                mb_old_log_probs = old_log_probs[i:end]
                mb_advantages = advantages[i:end]
                mb_raw_rewards = raw_rewards[i:end]
                mb_response_mask = response_mask[i:end]

                mb_loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=1,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=args.cliprange,
                )
                total_loss += mb_loss * (end - i)  # 按样本数加权

            avg_batch_loss = total_loss / total_samples  # 批次平均 loss
            avg_batch_loss.backward()  # 仅1次反向传播，无计算图销毁问题
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_pbar.set_postfix({"loss": f"{avg_batch_loss.item():.6f}"})

            # Update reference model periodically
            if (global_step + 1) % args.update_old_policy_freq == 0:
                # 同步 state_dict（跨设备安全）
                reference_model.load_state_dict(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
                reference_model.to(args.reference_model_device)
                reference_model.eval()

            global_step += 1

        # Save final model (Hugging Face 格式会自动处理 device_map)
        model.save_pretrained(args.checkpoint_path)
        tokenizer.save_pretrained(args.checkpoint_path)


# -----------------------------
# Parser with bool fix (same as before)
# -----------------------------


def get_rl_parser():
    parser = argparse.ArgumentParser(description="RL a Transformer model.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--sample_device", type=str, default="cuda:7")
    parser.add_argument("--reference_model_device", type=str, default="cuda:6")
    # 注意：不再需要 --rl_device
    parser.add_argument("--max_rl_gpu_memory_use", type=str, default="31GiB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_template_path", type=str, default="alignment/prompts/r1_zero.prompt"
    )
    parser.add_argument("--update_old_policy_freq", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=512)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="grpo_clip",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--rl_train_data", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--train_mini_batch_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument(
        "--use_std_normalization", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/math_rl")
    parser.add_argument("--rl_test_data", type=str, default="data/gsm8k/test.jsonl")

    return parser


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    parser = get_rl_parser()
    args = parser.parse_args()

    os.makedirs(args.checkpoint_path, exist_ok=True)

    checkpoint_exists = os.path.exists(args.checkpoint_path) and any(
        f.endswith(".bin") or f.endswith(".safetensors")
        for f in os.listdir(args.checkpoint_path)
    )
    if not checkpoint_exists:
        print("No checkpoint found, starting training...")
        train(args)
    else:
        print("Checkpoint found, skipping training...")

    print("Starting evaluation...")
    eval_model = LLM(model=args.checkpoint_path, dtype=args.dtype, seed=args.seed)
    evaluate_math(eval_model, args.prompt_template_path, args.rl_test_data)
