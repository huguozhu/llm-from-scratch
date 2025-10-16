import os
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from alignment.args import get_parser
from alignment.dataset import Gsm8kDataset
from torch.utils.data import DataLoader
from alignment.evaluate import evaluate_math
from accelerate import infer_auto_device_map, init_empty_weights


def init_vllm(
    model_id: str,
    device: str,
    dtype: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

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
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=dtype,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_device_ids(device: str) -> list:
    if device == "cpu":
        return []
    elif device.startswith("cuda"):
        # Handles formats like "cuda:0" and "cuda:0,cuda:1,cuda:2"
        device_parts = device.split(",")
        return [int(part.split(":")[1]) for part in device_parts]
    else:
        raise ValueError(f"Unknown device: {device}")


def evaluate_model(
    model: LLM, data_loader: DataLoader, tokenizer: AutoTokenizer
) -> float:
    """
    Evaluate the model on the given dataset.
    """
    for batch in data_loader:
        prompts, completions = batch
        outputs = model.generate(prompts, tokenizer)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            completion = output.outputs[0].text
            if i <= 2:
                print(f"Prompt: {prompt}\nCompletion: {completion}")


def sft():
    parser = get_parser()
    args = parser.parse_args()
    model_id = args.model
    eval_model = init_vllm(
        model_id=model_id,
        device=args.eval_device,
        dtype=args.dtype,
        seed=args.seed,
    )

    train_dataset = Gsm8kDataset(
        data_path=args.sft_train_data, promt_template_path=args.prompt_template_path
    )
    # test_dataset = Gsm8kDataset(
    #     data_path=args.sft_test_data, promt_template_path=args.prompt_template_path
    # )

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    # test_data_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False
    # )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Tie the weights before calculating the device map.
    empty_model.tie_weights()

    sft_device_ids = get_device_ids(args.sft_device)
    # You can adjust the memory usage per GPU here if needed
    max_memory = {dev_id: args.max_sft_gpu_memory_use for dev_id in sft_device_ids}

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        dtype=args.dtype,
        no_split_module_classes=empty_model._no_split_modules,
    )

    del empty_model

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    model.train()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_data_loader):
            prompts, completions, _ = batch

            full_texts = [
                p + c + tokenizer.eos_token for p, c in zip(prompts, completions)
            ]

            inputs = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            ).to(model.device)

            prompt_tokens = tokenizer(list(prompts), add_special_tokens=False)
            prompt_lengths = [len(ids) for ids in prompt_tokens.input_ids]

            labels = inputs.input_ids.clone()
            for idx in range(len(prompts)):
                prompt_len = prompt_lengths[idx]
                labels[idx, :prompt_len] = -100

            # Mask padding tokens
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Normalize the loss for accumulation
            loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Update weights only after accumulating gradients for N steps
            if (i + 1) % args.gradient_accumulation_steps == 0:
                print(
                    f"Epoch {epoch}, Iteration {i}, Loss: {loss.item() * args.gradient_accumulation_steps}"
                )
                optimizer.step()
                optimizer.zero_grad()

        # Evaluate the model on the test set
        load_policy_into_vllm_instance(model, eval_model)
        evaluate_math(
            eval_model,
            args.prompt_template_path,
            args.sft_test_data,
            args.batch_size,
            log_sample=True,
        )

        model.save_pretrained(args.checkpoint_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if (
        os.path.exists(args.checkpoint_path)
        and len(os.listdir(args.checkpoint_path)) > 0
    ):
        eval_model = LLM(
            model=args.checkpoint_path,
            tokenizer=args.model,  # Use the base model ID for the tokenizer
            trust_remote_code=True,
            dtype=args.dtype,
        )
        evaluate_math(
            eval_model, args.prompt_template_path, args.sft_test_data, log_sample=True
        )

    else:
        sft()
        model = LLM(args.checkpoint_path)
        evaluate_math(
            model, args.prompt_template_path, args.sft_test_data, log_sample=True
        )
