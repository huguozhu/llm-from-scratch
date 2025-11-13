import argparse


def get_sft_parser():
    """
    Returns the argument parser for SFT (Supervised Fine-Tuning).
    """
    parser = argparse.ArgumentParser(description="SFT a Transformer model.")

    # Model Hyperparameters
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="hf model id"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="dtype for training, bfloat16 by default, if bfloat16 is not supported in your device try float32 first, float16 is most-likely fail in old device",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Max sequence length"
    )

    # Optimizer Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Physical batch size for a single forward/backward pass. The effective batch size is 'batch_size * gradient_accumulation_steps'.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients over",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sft_device",
        type=str,
        default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6",
        help="Device to train on (e.g., 'cpu', 'cuda:0', 'mps')",
    )
    parser.add_argument(
        "--eval_device",
        type=str,
        default="cuda:7",
        help="Device to evaluate on (e.g., 'cpu', 'cuda:0', 'mps')",
    )
    parser.add_argument(
        "--max_sft_gpu_memory_use",
        type=str,
        default="31GiB",
        help="max gpu memory use per card for sft training",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/math_sft",
        help="Path to save checkpoints",
    )

    # Data Hyperparameters
    parser.add_argument(
        "--sft_train_data",
        type=str,
        default="data/gsm8k/train.jsonl",
        help="Path to the sft training data file",
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default="alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument(
        "--sft_test_data",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to the sft test data",
    )

    return parser


def get_rl_parser():
    """
    Returns the argument parser for RL (Reinforcement Learning).
    """
    parser = argparse.ArgumentParser(description="RL a Transformer model.")

    # Model Hyperparameters
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for training. 'bfloat16' is default. If not supported, try 'float32'. 'float16' may fail on older devices.",
    )
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=3,
        help="Min sequence length for sampling",
    )
    parser.add_argument(
        "--sample_device",
        type=str,
        default="cuda:7",
        help="Device for sampling from the model",
    )
    parser.add_argument(
        "--reference_model_device",
        type=str,
        default="cuda:6",
        help="Device for the reference model",
    )
    parser.add_argument(
        "--max_rl_gpu_memory_use",
        type=str,
        default="31GiB",
        help="Maximum GPU memory usage per card for RL training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default="alignment/prompts/r1_zero.prompt",
        help="Path to the prompt template",
    )
    parser.add_argument(
        "--update_old_policy_freq",
        type=int,
        default=10,
        help="Frequency of updating the old policy",
    )

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )
    parser.add_argument(
        "--advantage_eps",
        type=float,
        default=1e-6,
        help="Epsilon for advantage normalization",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=256,
        help="Batch size for rollouts",
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--sampling_min_tokens",
        type=int,
        default=4,
        help="Minimum number of tokens to sample",
    )
    parser.add_argument(
        "--sampling_max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to sample",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="grpo_clip",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        help="Type of loss function to use",
    )
    parser.add_argument(
        "--cliprange", type=float, default=0.2, help="Clip range for PPO"
    )
    parser.add_argument(
        "--rl_train_data",
        type=str,
        default="data/gsm8k/train.jsonl",
        help="Path to the RL training data",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Training batch size"
    )
    parser.add_argument("--group_size", type=int, default=4, help="Group size for GRPO")
    parser.add_argument(
        "--train_mini_batch_size",
        type=int,
        default=1,
        help="Training mini-batch size",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )

    # Normalization and Checkpointing
    parser.add_argument(
        "--use_std_normalization",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use standard deviation for normalization",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/math_rl",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--rl_test_data",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to the RL test data",
    )

    return parser
