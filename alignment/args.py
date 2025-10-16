import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="SFT a Transformer model.")

    # Model Hyperparameters
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="hf model id"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
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
