from collections.abc import Callable
from vllm import LLM, SamplingParams
import os
import json
from .r1_prompt import R1PromptTemplate

from .drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    log_sample: bool,
) -> dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    batch_size = len(prompts)
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    if log_sample and outputs:
        output = outputs[0]
        print("#" * 30)
        print("sample response:\n")
        print(f"Prompt: {output.prompt}")
        print(f"Completion: {output.outputs[0].text}")
        print(f"Ground Truth: {ground_truths[0]}")
        print("#" * 30)
    generated_texts = [output.outputs[0].text for output in outputs]
    rewards = [
        reward_fn(generated_text, ground_truth)
        for generated_text, ground_truth in zip(generated_texts, ground_truths)
    ]

    avg_format_rewards = sum([r["format_reward"] for r in rewards]) / batch_size
    avg_answer_rewards = sum([r["answer_reward"] for r in rewards]) / batch_size
    avg_all_rewards = sum([r["reward"] for r in rewards]) / batch_size
    print(f"avg_format_rewards: {avg_format_rewards}")
    print(f"avg_answer_rewards: {avg_answer_rewards}")
    print(f"avg_all_rewards: {avg_all_rewards}")
    return {
        "avg_format_rewards": avg_format_rewards,
        "avg_answer_rewards": avg_answer_rewards,
        "avg_all_rewards": avg_all_rewards,
    }


def get_gsm8k_test_data(test_data_path: os.PathLike) -> list[dict]:
    data = []
    with open(test_data_path, "r") as f:
        for line in f.readlines():
            obj = json.loads(line.strip())
            ts = obj["answer"].split("####")
            if len(ts) != 2:
                print(f"invalid answer: {obj['answer']}")
                continue
            data.append(
                {
                    "question": obj["question"],
                    "think": ts[0].strip(),
                    "answer": ts[1].strip(),
                }
            )
    return data


def evaluate_math(
    model: LLM,
    promt_path: os.PathLike,
    test_data_path: os.PathLike,
    batch_size: int = 128,
    log_sample=False,
):
    # sampling_params = SamplingParams(
    #     temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    # )
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)

    test_data = get_gsm8k_test_data(test_data_path)
    R1PT = R1PromptTemplate(promt_path)

    promts = [R1PT.gen_prompt(d["question"]) for d in test_data]
    ground_truths = [d["answer"] for d in test_data]

    format_rewards = []
    answer_rewards = []
    all_rewards = []
    for i in range(len(test_data) // batch_size):
        print(f"evaluate batch {i}/{len(test_data) // batch_size}")
        batch_promts = promts[i * batch_size : (i + 1) * batch_size]
        batch_ground_truth = ground_truths[i * batch_size : (i + 1) * batch_size]
        rewards = evaluate_vllm(
            model,
            r1_zero_reward_fn,
            batch_promts,
            batch_ground_truth,
            sampling_params,
            log_sample,
        )
        format_rewards.append(rewards["avg_format_rewards"])
        answer_rewards.append(rewards["avg_answer_rewards"])
        all_rewards.append(rewards["avg_all_rewards"])
    print("*" * 30)
    print("Final AVERAGE Rewards:\n")
    print(f"avg_format_rewards: {sum(format_rewards) / len(format_rewards)}")
    print(f"avg_answer_rewards: {sum(answer_rewards) / len(answer_rewards)}")
    print(f"avg_all_rewards: {sum(all_rewards) / len(all_rewards)}")


if __name__ == "__main__":
    promt_path = "alignment/prompts/r1_zero.prompt"
    test_data_path = "data/gsm8k/test.jsonl"
    model = LLM(model="Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True, dtype="float16")

    evaluate_math(model, promt_path, test_data_path, log_sample=True)
