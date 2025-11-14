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
    Compute group-normalized rewards (advantages) for GRPO.

    Each group of `group_size` responses shares the same prompt (hence same ground truth).
    Advantages are computed as: (reward - group_mean) / (group_std + eps)

    Args:
        reward_fn: Function that takes (response, ground_truth) and returns a dict with "reward".
        rollout_responses: List of model responses. Length = B = n_prompts * group_size.
        repeated_ground_truths: Ground truths repeated per response. Same length as rollout_responses.
        group_size: Number of responses per prompt/group.
        advantage_eps: Small constant added to std for numerical stability.
        normalize_by_std: If True, divide by group std; otherwise, only center by mean.

    Returns:
        - normalized_rewards: [B] tensor of group-normalized advantages (flat).
        - raw_rewards: [B] tensor of original rewards.
        - meta: Dictionary containing "mean_advantages" and optionally "std_advantages".
    """
    # Validate input length
    assert len(rollout_responses) == len(repeated_ground_truths), (
        "rollout_responses and repeated_ground_truths must have the same length"
    )
    assert len(rollout_responses) % group_size == 0, (
        "Total number of responses must be divisible by group_size"
    )

    # Compute raw rewards
    rewards = [
        reward_fn(resp, gt)
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    reward_values = [r["reward"] for r in rewards]
    raw_rewards = torch.tensor(reward_values, dtype=torch.float32)

    # Reshape into groups: [n_prompts, group_size]
    advantages = raw_rewards.view(-1, group_size)

    # Subtract group mean
    mean_advantages = einx.mean("n_prompts group_size -> n_prompts 1", advantages)
    advantages = advantages - mean_advantages

    meta: Dict[str, Tensor] = {"mean_advantages": mean_advantages}

    # Optional: divide by group std
    if normalize_by_std:
        # var = einx.mean("n_prompts group_size -> n_prompts 1", advantages**2)
        std_advantages = torch.std(advantages, dim=1, unbiased=True, keepdim=True)
        advantages = advantages / (std_advantages + advantage_eps)
        meta["std_advantages"] = std_advantages

    advantages = advantages.view(-1)

    return advantages, raw_rewards, meta


def compute_naive_policy_gradient_loss(
    advantages_or_raw_reward: torch.Tensor,  ## [batch_size, 1]
    policy_log_probs: torch.Tensor,  ## [batch_size, seq_len]
) -> torch.Tensor:  ## [batch_size, seq_len], per_token policy_gradient loss
    """
    Compute naive policy gradient loss for GRPO.

    Args:
        advantages: [B] tensor of raw advantages.
        policy_log_probs: [B, S] tensor of log probabilities of actions under current policy.

    Returns:
        - loss: Scalar tensor of negative mean policy gradient loss.
    """
    if advantages_or_raw_reward.dim() == 1:
        advantages_or_raw_reward = advantages_or_raw_reward.unsqueeze(-1)
    return -advantages_or_raw_reward * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,  ## [batch_size, 1]
    policy_log_probs: torch.Tensor,  ## [batch_size, seq_len]
    old_log_probs: torch.Tensor,  ## [batch_size, seq_len]
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute GRPO clip loss for policy gradient.

    Args:
        advantages: [B] tensor of raw advantages.
        policy_log_probs: [B, S] tensor of log probabilities of actions under current policy.
        old_log_probs: [B, S] tensor of log probabilities of actions under old policy.
        cliprange: Clip range for policy gradient.

    Returns:
        - loss: Scalar tensor of negative mean policy gradient loss.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    # important：detach old_log_probs to avoid update
    old_log_probs = old_log_probs.detach()

    log_ratio = policy_log_probs - old_log_probs
    importance = torch.exp(log_ratio)
    clip_importance = torch.clamp(importance, 1 - cliprange, 1 + cliprange)

    adv = importance * advantages
    clip_adv = clip_importance * advantages

    meta = {}
    meta["clip"] = adv <= clip_adv

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
    Compute policy gradient loss for GRPO.

    Returns:
        - loss: Scalar tensor (mean over valid tokens and batch).
        - meta: diagnostics dict.
    """
    # Parameter validation
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for 'no_baseline'"
        per_token_loss = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs
        )
        meta = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, (
            "advantages required for 'reinforce_with_baseline'"
        )
        per_token_loss = compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        )
        meta = {}
    else:  # grpo_clip
        assert advantages is not None, "advantages required for 'grpo_clip'"
        assert old_log_probs is not None, "old_log_probs required for 'grpo_clip'"
        assert cliprange is not None, "cliprange required for 'grpo_clip'"
        per_token_loss, meta = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    return per_token_loss, meta


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, eps=1e-8
) -> torch.Tensor:
    """
    Compute mean of tensor along given dimension, masked by boolean mask.
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
) -> tuple[torch.Tensor, dict[str, Tensor]]:
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mean_loss = masked_mean(per_token_loss, response_mask)

    if gradient_accumulation_steps > 1:
        mean_loss = mean_loss / gradient_accumulation_steps

    mean_loss.backward()

    return mean_loss.detach(), meta
