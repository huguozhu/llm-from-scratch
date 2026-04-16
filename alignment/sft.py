# ==============================================================================
# 监督微调（SFT）训练模块
# ==============================================================================
# 功能概述：
#   在预训练模型（如 Qwen2.5-Math-1.5B）上进行监督微调。
#   以 GSM8K 数学推理数据集为训练源，教会模型生成结构化推理过程。
#   关键特性：梯度累积（等效大 batch）、bfloat16/float32 精度训练、
#   交叉熵损失仅计算 answer 部分（prompt mask 为 -100）。
# ==============================================================================
import os
import torch
from vllm import LLM
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from alignment.args import get_sft_parser
from alignment.dataset import Gsm8kDataset
from torch.utils.data import DataLoader
from alignment.evaluate import evaluate_math
from alignment.util import get_device_map, init_vllm


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


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
    parser = get_sft_parser()
    args = parser.parse_args()
    model_id = args.model
    eval_model = init_vllm(
        model_path=model_id,
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

    device_map = get_device_map(
        args.model, args.sft_device, args.max_sft_gpu_memory_use, args.dtype
    )

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
        tokenizer.save_pretrained(args.checkpoint_path)


if __name__ == "__main__":
    parser = get_sft_parser()
    args = parser.parse_args()
    if (
        os.path.exists(args.checkpoint_path)
        and len(os.listdir(args.checkpoint_path)) > 0
    ):
        model = LLM(args.checkpoint_path)
        evaluate_math(model, args.prompt_template_path, args.sft_test_data)

    else:
        sft()
        model = LLM(args.checkpoint_path, tokenizer=args.model)
        evaluate_math(model, args.prompt_template_path, args.sft_test_data)
