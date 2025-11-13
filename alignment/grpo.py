from typing import Callable, Dict, List, Literal
import torch
from torch import Tensor
import einx


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    Computes group-normalized rewards (advantages) for GRPO.

    Each group of `group_size` responses shares the same prompt and, therefore, the same ground truth.
    The advantages are calculated as: (reward - group_mean) / (group_std + eps).

    Args:
        reward_fn: A function that takes a response and a ground truth, and returns a dict with a "reward" key.
        rollout_responses: A list of model responses. The length is B = n_prompts * group_size.
        repeated_ground_truths: A list of ground truths, repeated for each response. It has the same length as `rollout_responses`.
        group_size: The number of responses per prompt/group.
        advantage_eps: A small constant added to the standard deviation for numerical stability.
        normalize_by_std: If True, the rewards are divided by the group's standard deviation. Otherwise, they are only centered by the mean.

    Returns:
        - normalized_rewards: A tensor of shape [B] containing the group-normalized advantages (flattened).
        - raw_rewards: A tensor of shape [B] containing the original rewards.
        - meta: A dictionary containing "mean_advantages" and, optionally, "std_advantages".
    """
    # Validate that the lengths of the inputs are consistent.
    assert len(rollout_responses) == len(repeated_ground_truths), (
        "rollout_responses and repeated_ground_truths must have the same length"
    )
    assert len(rollout_responses) % group_size == 0, (
        "The total number of responses must be divisible by group_size"
    )

    # --- 1. Compute raw rewards ---
    rewards = [
        reward_fn(resp, gt)
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    reward_values = [r["reward"] for r in rewards]
    raw_rewards = torch.tensor(reward_values, dtype=torch.float32)

    # --- 2. Reshape rewards into groups ---
    # The shape becomes [n_prompts, group_size].
    advantages = raw_rewards.view(-1, group_size)

    # --- 3. Center rewards by subtracting the group mean ---
    mean_advantages = einx.mean("n_prompts group_size -> n_prompts 1", advantages)
    advantages = advantages - mean_advantages

    meta: Dict[str, Tensor] = {"mean_advantages": mean_advantages}

    # --- 4. Optionally, normalize by the group standard deviation ---
    if normalize_by_std:
        std_advantages = torch.std(advantages, dim=1, unbiased=True, keepdim=True)
        advantages = advantages / (std_advantages + advantage_eps)
        meta["std_advantages"] = std_advantages

    # --- 5. Flatten the advantages back to a 1D tensor ---
    advantages = advantages.view(-1)

    return advantages, raw_rewards, meta


def compute_naive_policy_gradient_loss(
    advantages_or_raw_reward: torch.Tensor,  # [batch_size, 1]
    policy_log_probs: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:  # [batch_size, seq_len], per-token policy gradient loss
    """
    Computes the naive policy gradient loss for GRPO.

    Args:
        advantages_or_raw_reward: A tensor of shape [B] or [B, 1] containing the raw advantages or rewards.
        policy_log_probs: A tensor of shape [B, S] containing the log probabilities of actions under the current policy.

    Returns:
        A tensor of shape [B, S] representing the per-token policy gradient loss.
    """
    if advantages_or_raw_reward.dim() == 1:
        advantages_or_raw_reward = advantages_or_raw_reward.unsqueeze(-1)
    return -advantages_or_raw_reward * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,  # [batch_size, 1]
    policy_log_probs: torch.Tensor,  # [batch_size, seq_len]
    old_log_probs: torch.Tensor,  # [batch_size, seq_len]
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the GRPO clip loss for the policy gradient.

    Args:
        advantages: A tensor of shape [B] or [B, 1] containing the raw advantages.
        policy_log_probs: A tensor of shape [B, S] containing the log probabilities of actions under the current policy.
        old_log_probs: A tensor of shape [B, S] containing the log probabilities of actions under the old policy.
        cliprange: The clip range for the policy gradient.

    Returns:
        - A tensor of shape [B, S] representing the per-token GRPO clip loss.
        - A metadata dictionary containing a "clip" mask.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    # --- 1. Compute the importance ratio ---
    log_ratio = policy_log_probs - old_log_probs
    importance = torch.exp(log_ratio)

    # --- 2. Clip the importance ratio ---
    clip_importance = torch.clamp(importance, 1 - cliprange, 1 + cliprange)

    # --- 3. Compute the unclipped and clipped advantages ---
    adv = importance * advantages
    clip_adv = clip_importance * advantages

    # --- 4. Create metadata for diagnostics ---
    meta = {"clip": adv <= clip_adv}

    # --- 5. Return the minimum of the two advantage types, negated ---
    return -torch.min(adv, clip_adv), meta


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,  # [B, L]
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,  # [B]
    advantages: torch.Tensor | None = None,  # [B]
    old_log_probs: torch.Tensor | None = None,  # [B, L]
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the policy gradient loss for GRPO, delegating to the appropriate loss function.

    Args:
        policy_log_probs: Log probabilities of the policy.
        loss_type: The type of loss to compute.
        raw_rewards: Raw rewards, required for 'no_baseline'.
        advantages: Advantages, required for 'reinforce_with_baseline' and 'grpo_clip'.
        old_log_probs: Log probabilities of the old policy, required for 'grpo_clip'.
        cliprange: Clip range, required for 'grpo_clip'.

    Returns:
        - A tensor representing the per-token loss.
        - A dictionary with diagnostic metadata.
    """
    # --- 1. Validate parameters based on loss_type ---
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for 'no_baseline' loss"
        per_token_loss = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs
        )
        meta = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, (
            "advantages is required for 'reinforce_with_baseline' loss"
        )
        per_token_loss = compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        )
        meta = {}
    else:  # grpo_clip
        assert advantages is not None, "advantages is required for 'grpo_clip' loss"
        assert old_log_probs is not None, "old_log_probs is required for 'grpo_clip' loss"
        assert cliprange is not None, "cliprange is required for 'grpo_clip' loss"
        per_token_loss, meta = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    return per_token_loss, meta


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, eps=1e-8
) -> torch.Tensor:
    """
    Computes the mean of a tensor along a given dimension, masked by a boolean mask.
    """
    mask = mask.float()
    masked_tensor = tensor * mask
    filled_sum = masked_tensor.sum(dim)
    return filled_sum / (mask.sum(dim) + eps)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Performs a single microbatch training step for GRPO.

    Args:
        policy_log_probs: Log probabilities of the policy.
        response_mask: A mask for the response tokens.
        gradient_accumulation_steps: The number of steps to accumulate gradients over.
        loss_type: The type of loss to compute.
        raw_rewards: Raw rewards, required for 'no_baseline'.
        advantages: Advantages, required for 'reinforce_with_baseline' and 'grpo_clip'.
        old_log_probs: Log probabilities of the old policy, required for 'grpo_clip'.
        cliprange: Clip range, required for 'grpo_clip'.

    Returns:
        - The mean loss for the microbatch.
        - A dictionary with diagnostic metadata.
    """
    # --- 1. Compute the per-token policy gradient loss ---
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # --- 2. Compute the masked mean loss ---
    mean_loss = masked_mean(per_token_loss, response_mask)

    # --- 3. (Optional) Normalize and backpropagate the loss ---
    # mean_loss = mean_loss / gradient_accumulation_steps
    # mean_loss.backward()

    return mean_loss, meta
