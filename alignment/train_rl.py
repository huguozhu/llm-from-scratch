# ==============================================================================
# 强化学习训练主循环
# ==============================================================================
# 功能概述：
#   使用 GRPO 算法对 SFT 后的模型做强化学习训练，提升数学推理能力。
#   流程：加载 SFT 检查点 -> 采样问题 -> 生成 G 个候选回答 -> 计算奖励 ->
#   GRPO 策略梯度更新 -> KL 正则化 -> 定期 GSM8K 评估。
# ==============================================================================
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from vllm import SamplingParams, LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import math
import gc

from alignment.args import get_rl_parser, get_sft_parser
from alignment.dataset import Gsm8kDataset
from alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from alignment.sft import init_vllm
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.evaluate import evaluate_math
from alignment.util import str_to_torch_dtype
from torch.utils.tensorboard.writer import SummaryWriter


def partition_model_across_devices(args) -> dict[str, int]:
    total_gpu_count = torch.cuda.device_count()
    if total_gpu_count < 4:
        raise ValueError(
            "This script requires at least 3 GPUs: 1 for sampling, 1 for reference, and at least 1 for the policy model."
        )

    try:
        sample_device_idx = int(args.sample_device.split(":")[-1])
        ref_device_idx = int(args.reference_model_device.split(":")[-1])
        eval_device_idx = int(args.eval_device.split(":")[-1])
    except (ValueError, IndexError):
        raise ValueError(
            "Invalid device format. Please use 'cuda:N' for device arguments."
        )

    reserved_indices = {sample_device_idx, ref_device_idx, eval_device_idx}
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
    print(f"Reserved for eval model: cuda:{eval_device_idx}")
    print(f"MANUALLY balancing policy model across: {policy_devices_str}")
    print("-" * 60)

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
    return device_map


def train(args, base_model_checkpoint_path: os.PathLike):
    writer = SummaryWriter(log_dir=args.log_dir)
    dtype = str_to_torch_dtype(args.dtype)
    group_size = 1 if args.loss_type == "no_baseline" else args.group_size

    # === Step 1: Initialize vLLM for sampling (rollout) ===
    print(f"Initializing vLLM sample_model on {args.sample_device}...")
    sample_model = init_vllm(
        model_path=base_model_checkpoint_path,
        tokenizer_path=args.model,
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

    device_map = partition_model_across_devices(args)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_checkpoint_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )

    model.train()
    policy_device = model.device

    # === Step 3: Load reference model (frozen) on its dedicated GPU ===
    print(f"Loading reference_model on {args.reference_model_device}...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        base_model_checkpoint_path,
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
    while global_step < args.total_trainging_steps:
        for batch_id, batch in enumerate(train_data_loader):
            prompts, _, ground_truths = batch

            grouped_prompts = [p for p in prompts for _ in range(group_size)]
            grouped_ground_truths = [
                gt for gt in ground_truths for _ in range(group_size)
            ]

            # === Rollout with vLLM ===
            sampling_params = SamplingParams(
                temperature=args.sampling_temperature,
                top_p=args.sampling_top_p,
                min_tokens=args.sampling_min_tokens,
                max_tokens=args.sampling_max_tokens,
                stop=["</answer>"],
                repetition_penalty=1.1,
                include_stop_str_in_output=True,
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

            # === Tokenize all data for the batch ===
            prompt_tokens = tokenizer(
                grouped_prompts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            )
            full_tokens = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            )

            # === Prepare for Gradient Accumulation ===
            total_samples = len(full_texts)
            total_loss = 0.0
            optimizer.zero_grad()

            # Calculate the number of gradient accumulation steps
            grad_acc_steps = math.ceil(total_samples / args.train_mini_batch_size)

            # The main training loop now iterates over microbatches.
            # To correctly implement gradient accumulation, the forward pass and all
            # subsequent calculations (log probs, loss, etc.) must be *inside* this
            # loop. This ensures that a new computational graph is created for each
            # microbatch. Calling .backward() on the loss from each microbatch then
            # accumulates the gradients correctly without causing a "trying to backward
            # through the graph a second time" error.
            for i in range(0, total_samples, args.train_mini_batch_size):
                end = min(i + args.train_mini_batch_size, total_samples)

                # --- Microbatch Slicing ---
                mb_input_ids = full_tokens.input_ids[i:end].to(policy_device)
                mb_attention_mask = full_tokens.attention_mask[i:end].to(policy_device)
                mb_prompt_lengths = prompt_tokens.attention_mask.sum(dim=1)[i:end].to(
                    policy_device
                )
                mb_advantages = advantages[i:end].to(policy_device)
                mb_raw_rewards = raw_rewards[i:end].to(policy_device)

                # --- Forward Pass for Microbatch ---
                policy_logits = model(
                    input_ids=mb_input_ids, attention_mask=mb_attention_mask
                ).logits

                with torch.inference_mode():
                    ref_input_ids = mb_input_ids.to(args.reference_model_device)
                    ref_attention_mask = mb_attention_mask.to(
                        args.reference_model_device
                    )
                    old_logits = reference_model(
                        input_ids=ref_input_ids, attention_mask=ref_attention_mask
                    ).logits
                old_logits = old_logits.to(policy_device)

                # --- Log Probs for Microbatch ---
                policy_log_probs_all = torch.log_softmax(policy_logits, dim=-1)
                old_log_probs_all = torch.log_softmax(old_logits, dim=-1)

                policy_log_probs = torch.gather(
                    policy_log_probs_all, -1, mb_input_ids.unsqueeze(-1)
                ).squeeze(-1)

                old_log_probs = (
                    torch.gather(old_log_probs_all, -1, mb_input_ids.unsqueeze(-1))
                    .squeeze(-1)
                    .detach()
                )

                # --- Response Mask for Microbatch ---
                response_mask = torch.zeros_like(mb_input_ids, dtype=torch.bool)
                for j in range(len(response_mask)):
                    start = mb_prompt_lengths[j].item()
                    # Use attention mask sum for the end to handle padding correctly
                    end_pos = mb_attention_mask[j].sum().item()
                    response_mask[j, start:end_pos] = True

                response_mask &= mb_input_ids != tokenizer.pad_token_id
                if tokenizer.eos_token_id is not None:
                    response_mask &= mb_input_ids != tokenizer.eos_token_id
                if tokenizer.bos_token_id is not None:
                    response_mask &= mb_input_ids != tokenizer.bos_token_id

                # --- GRPO Train Step (includes backward pass) ---
                mb_loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=grad_acc_steps,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=old_log_probs,
                    cliprange=args.cliprange,
                )

                total_loss += mb_loss.item() * (end - i)

            # === Optimizer Step after Gradient Accumulation ===
            avg_batch_loss = total_loss / total_samples
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            print(
                {
                    f"Step {global_step + 1}/{args.total_trainging_steps} loss": f"{avg_batch_loss:.6f}"
                }
            )

            if (global_step + 1) % args.evaluate_freq == 0:
                model.save_pretrained(args.tmp_checkpoint_path)
                tokenizer.save_pretrained(args.tmp_checkpoint_path)

                eval_model = LLM(
                    model=args.tmp_checkpoint_path,
                    dtype=args.dtype,
                    seed=args.seed,
                    device=args.eval_device,
                )
                score = evaluate_math(
                    eval_model,
                    args.prompt_template_path,
                    args.rl_test_data,
                    log_sample=True,
                    # num_test_batches=5,
                )
                print(
                    f"{global_step + 1}/{args.total_trainging_steps} eval score: {score}"
                )
                # This re-initialization might be unnecessary/inefficient depending on workflow
                # For now, keeping it as it was in the original code.
                # reference_model = LLM(
                #     model=args.ref_model_checkpoint_path,
                #     dtype=args.dtype,
                #     seed=args.seed,
                # )
                writer.add_scalar(
                    "eval_score", score["avg_all_rewards"], global_step + 1
                )
                del eval_model
                torch.cuda.empty_cache()
                gc.collect()

            # Update reference model periodically
            # if (global_step + 1) % args.update_old_policy_freq == 0:
            #     # Safely transfer state_dict across devices
            #     reference_model.load_state_dict(
            #         {k: v.cpu() for k, v in model.state_dict().items()}
            #     )
            #     reference_model.to(args.reference_model_device)
            #     reference_model.eval()

            global_step += 1
            if global_step >= args.total_trainging_steps:
                break

        model.save_pretrained(args.checkpoint_path)
        tokenizer.save_pretrained(args.checkpoint_path)


if __name__ == "__main__":
    parser = get_rl_parser()
    args = parser.parse_args()

    sft_parser = get_sft_parser()
    sft_args = sft_parser.parse_args()

    os.makedirs(args.checkpoint_path, exist_ok=True)

    checkpoint_exists = os.path.exists(args.checkpoint_path) and any(
        f.endswith(".bin") or f.endswith(".safetensors")
        for f in os.listdir(args.checkpoint_path)
    )
    if not checkpoint_exists:
        print("No checkpoint found, starting training...")
        train(args, sft_args.checkpoint_path)
    else:
        print("Checkpoint found, skipping training...")

    print("Starting evaluation...")
    eval_model = LLM(model=args.checkpoint_path, dtype=args.dtype, seed=args.seed)
    evaluate_math(
        eval_model, args.prompt_template_path, args.rl_test_data, log_sample=True
    )
