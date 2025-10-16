import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.nn.utils.rnn as rnn_utils

import torch.nn.functional as F


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
