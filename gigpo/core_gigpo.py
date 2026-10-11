# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any, Optional, Tuple


"""
Core functions to implement the GiGPO algorithm (https://arxiv.org/abs/2505.10978).
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    summary = build_group_size_summary(group_size)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")


def build_group_size_summary(group_size: Sequence[int]) -> Dict[int, Tuple[int, float]]:
    """Return ``{group_size: (count, proportion_of_groups)}``."""
    normalized_sizes = [int(size) for size in group_size if int(size) > 0]
    if not normalized_sizes:
        return {}

    counts = Counter(normalized_sizes)
    total = sum(counts.values())
    max_size = max(counts)

    summary: Dict[int, Tuple[int, float]] = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0.0
        summary[size] = (cnt, prop)
    return summary


def compute_group_size_distribution_metrics(group_size: Sequence[int], prefix: str) -> Dict[str, float]:
    """
    Flatten a group-size distribution into scalar metrics that can be logged per
    training step.
    """
    max_explicit_bucket = 10
    normalized_sizes = [int(size) for size in group_size if int(size) > 0]
    bucket_labels = [str(size) for size in range(1, max_explicit_bucket + 1)] + [f"gt_{max_explicit_bucket}"]

    if not normalized_sizes:
        metrics = {
            f"{prefix}/num_groups": 0.0,
            f"{prefix}/num_samples": 0.0,
            f"{prefix}/mean": 0.0,
            f"{prefix}/std": 0.0,
            f"{prefix}/min": 0.0,
            f"{prefix}/max": 0.0,
        }
        for label in bucket_labels:
            metrics[f"{prefix}/size_{label}_group_count"] = 0.0
            metrics[f"{prefix}/size_{label}_group_prop"] = 0.0
            metrics[f"{prefix}/size_{label}_sample_prop"] = 0.0
        return metrics

    total_groups = float(len(normalized_sizes))
    total_samples = float(sum(normalized_sizes))
    metrics = {
        f"{prefix}/num_groups": total_groups,
        f"{prefix}/num_samples": total_samples,
        f"{prefix}/mean": float(np.mean(normalized_sizes)),
        f"{prefix}/std": float(np.std(normalized_sizes)),
        f"{prefix}/min": float(np.min(normalized_sizes)),
        f"{prefix}/max": float(np.max(normalized_sizes)),
    }

    bucket_group_counts = {label: 0 for label in bucket_labels}
    bucket_sample_totals = {label: 0 for label in bucket_labels}
    for size in normalized_sizes:
        label = str(size) if size <= max_explicit_bucket else f"gt_{max_explicit_bucket}"
        bucket_group_counts[label] += 1
        bucket_sample_totals[label] += size

    for label in bucket_labels:
        cnt = float(bucket_group_counts[label])
        metrics[f"{prefix}/size_{label}_group_count"] = cnt
        metrics[f"{prefix}/size_{label}_group_prop"] = cnt / total_groups if total_groups > 0 else 0.0
        metrics[f"{prefix}/size_{label}_sample_prop"] = float(bucket_sample_totals[label]) / total_samples if total_samples > 0 else 0.0
    return metrics


def compute_step_group_distribution_metrics(step_group_uids: Sequence, prefix: str) -> Dict[str, float]:
    """Compute group-size metrics from the raw step-group assignments."""
    if len(step_group_uids) == 0:
        return compute_group_size_distribution_metrics([], prefix=prefix)

    group_counts = Counter(step_group_uids.tolist() if isinstance(step_group_uids, np.ndarray) else list(step_group_uids))
    return compute_group_size_distribution_metrics(group_counts.values(), prefix=prefix)


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns


def _mode_to_remove_std(mode: str) -> bool:
    """Map normalization mode to whether std normalization should be removed."""
    if mode == "mean_std_norm":
        return False
    if mode == "mean_norm":
        return True
    raise ValueError(f"Unknown mode: {mode}")


def _broadcast_mask_to_response(
    mask: Optional[torch.Tensor | np.ndarray | Sequence],
    response_mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    Broadcast a sample-level or token-level mask to match ``response_mask``.
    """
    if mask is None:
        return torch.ones_like(response_mask, dtype=response_mask.dtype)

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    elif not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    mask = mask.to(device=response_mask.device)

    if mask.dim() == 1:
        if mask.shape[0] != response_mask.shape[0]:
            raise ValueError(
                f"{name} batch dimension mismatch: {mask.shape[0]} vs {response_mask.shape[0]}"
            )
        mask = mask.unsqueeze(-1).expand_as(response_mask)
    elif mask.dim() == 2:
        if mask.shape == response_mask.shape:
            pass
        elif mask.shape[0] == response_mask.shape[0] and mask.shape[1] == 1:
            mask = mask.expand_as(response_mask)
        else:
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} is not compatible with response_mask "
                f"shape {tuple(response_mask.shape)}"
            )
    else:
        raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(mask.shape)}")

    return mask.to(dtype=response_mask.dtype)


def compute_teacher_token_advantage(
    response_mask: torch.Tensor,
    teacher_log_prob: Optional[torch.Tensor] = None,
    old_log_prob: Optional[torch.Tensor] = None,
    critical_step_mask: Optional[torch.Tensor | np.ndarray | Sequence] = None,
    normalize: bool = False,
    clip_range: Optional[float] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute token-level teacher advantage for SEED.

    The default teacher signal is the detached log-prob improvement under the
    enhanced prompt:
        ``teacher_log_prob - old_log_prob``.
    """
    if teacher_log_prob is None or old_log_prob is None:
        return torch.zeros(response_mask.shape, dtype=torch.float32, device=response_mask.device)

    if teacher_log_prob.shape != old_log_prob.shape:
        raise ValueError(
            f"teacher_log_prob shape {tuple(teacher_log_prob.shape)} does not match "
            f"old_log_prob shape {tuple(old_log_prob.shape)}"
        )
    if teacher_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"teacher_log_prob shape {tuple(teacher_log_prob.shape)} does not match "
            f"response_mask shape {tuple(response_mask.shape)}"
        )

    critical_mask = _broadcast_mask_to_response(
        mask=critical_step_mask,
        response_mask=response_mask,
        name="critical_step_mask",
    )

    teacher_advantages = (teacher_log_prob - old_log_prob).detach()
    teacher_advantages = teacher_advantages * response_mask.to(dtype=teacher_advantages.dtype)
    teacher_advantages = teacher_advantages * critical_mask.to(dtype=teacher_advantages.dtype)

    if normalize:
        valid_mask = (response_mask > 0) & (critical_mask > 0)
        if valid_mask.any():
            valid_values = teacher_advantages[valid_mask]
            mean = valid_values.mean()
            std = valid_values.std(unbiased=False)
            if std > epsilon:
                teacher_advantages = torch.where(
                    valid_mask,
                    (teacher_advantages - mean) / (std + epsilon),
                    teacher_advantages,
                )
            else:
                teacher_advantages = torch.where(
                    valid_mask,
                    teacher_advantages - mean,
                    teacher_advantages,
                )

    if clip_range is not None:
        teacher_advantages = torch.clamp(teacher_advantages, -clip_range, clip_range)

    return teacher_advantages


def summarize_seed_advantage_components(
    response_mask: torch.Tensor,
    episode_advantages: torch.Tensor,
    step_advantages: torch.Tensor,
    teacher_advantages: torch.Tensor,
    step_advantage_w: float = 0.0,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Summarize the relative contribution of the three SEED advantage terms.

    The reported share is based on the weighted absolute mean magnitude of each
    component over valid response tokens:
        episode_adv
        step_advantage_w * step_adv
        teacher_adv
    """
    valid_mask = response_mask > 0
    def _summarize(prefix: str, components: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if not valid_mask.any():
            return {
                f"{prefix}/episode_mean": 0.0,
                f"{prefix}/step_mean": 0.0,
                f"{prefix}/teacher_mean": 0.0,
                f"{prefix}/episode_abs_mean": 0.0,
                f"{prefix}/step_abs_mean": 0.0,
                f"{prefix}/teacher_abs_mean": 0.0,
                f"{prefix}/episode_share": 0.0,
                f"{prefix}/step_share": 0.0,
                f"{prefix}/teacher_share": 0.0,
                f"{prefix}/total_abs_mean": 0.0,
            }

        mean_metrics: Dict[str, float] = {}
        abs_mean_metrics: Dict[str, float] = {}
        for name, component in components.items():
            valid_values = component.detach()[valid_mask]
            mean_metrics[name] = float(valid_values.mean().cpu().item())
            abs_mean_metrics[name] = float(valid_values.abs().mean().cpu().item())

        total_abs_mean = sum(abs_mean_metrics.values())
        return {
            f"{prefix}/episode_mean": mean_metrics["episode"],
            f"{prefix}/step_mean": mean_metrics["step"],
            f"{prefix}/teacher_mean": mean_metrics["teacher"],
            f"{prefix}/episode_abs_mean": abs_mean_metrics["episode"],
            f"{prefix}/step_abs_mean": abs_mean_metrics["step"],
            f"{prefix}/teacher_abs_mean": abs_mean_metrics["teacher"],
            f"{prefix}/episode_share": abs_mean_metrics["episode"] / (total_abs_mean + epsilon),
            f"{prefix}/step_share": abs_mean_metrics["step"] / (total_abs_mean + epsilon),
            f"{prefix}/teacher_share": abs_mean_metrics["teacher"] / (total_abs_mean + epsilon),
            f"{prefix}/total_abs_mean": total_abs_mean,
        }

    raw_components = {
        "episode": episode_advantages,
        "step": step_advantages,
        "teacher": teacher_advantages,
    }
    weighted_components = {
        "episode": episode_advantages,
        "step": step_advantage_w * step_advantages,
        "teacher": teacher_advantages,
    }

    teacher_active_token_ratio = float(
        ((teacher_advantages.detach().abs() > 0) & valid_mask).float().sum().cpu().item()
        / (valid_mask.float().sum().cpu().item() + epsilon)
    ) if valid_mask.any() else 0.0

    metrics = {}
    metrics.update(_summarize("seed/adv_raw", raw_components))
    metrics.update(_summarize("seed/adv", weighted_components))
    metrics.update({
        "seed/adv/teacher_active_token_ratio": teacher_active_token_ratio,
        "seed/adv/step_weight": float(step_advantage_w),
    })
    return metrics


def compute_seed_advantage_components(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    teacher_log_prob: Optional[torch.Tensor] = None,
    episode_teacher_log_prob: Optional[torch.Tensor] = None,
    step_teacher_log_prob: Optional[torch.Tensor] = None,
    old_log_prob: Optional[torch.Tensor] = None,
    critical_step_mask: Optional[torch.Tensor | np.ndarray | Sequence] = None,
    step_skill_mask: Optional[torch.Tensor | np.ndarray | Sequence] = None,
    epsilon: float = 1e-6,
    step_advantage_w: float = 0.0,
    episode_skill_teacher_advantage_w: float = 1.0,
    step_skill_teacher_advantage_w: float = 0.0,
    mode: str = "mean_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    normalize_teacher_adv: bool = False,
    clip_teacher_adv: Optional[float] = None,
    metrics_prefix: str = "seed/state_group",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute SEED advantages with independently weighted episode- and step-skill teacher terms.
    """
    remove_std = _mode_to_remove_std(mode)

    episode_advantages = episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        epsilon=epsilon,
        remove_std=remove_std,
    )

    step_weight = float(step_advantage_w or 0.0)
    if step_weight != 0.0:
        if step_rewards is None:
            raise ValueError("step_rewards is required when algorithm.seed.step_advantage_w is non-zero.")
        step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)
        step_group_metrics = compute_step_group_distribution_metrics(
            step_group_uids=step_group_uids,
            prefix=metrics_prefix,
        )
        step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)
    else:
        step_advantages = torch.zeros_like(episode_advantages)
        step_group_metrics: Dict[str, float] = {}
    episode_skill_weight = float(episode_skill_teacher_advantage_w or 0.0)
    step_skill_weight = float(step_skill_teacher_advantage_w or 0.0)

    episode_teacher_advantages = compute_teacher_token_advantage(
        response_mask=response_mask,
        teacher_log_prob=episode_teacher_log_prob if episode_teacher_log_prob is not None else teacher_log_prob,
        old_log_prob=old_log_prob,
        critical_step_mask=critical_step_mask,
        normalize=normalize_teacher_adv,
        clip_range=clip_teacher_adv,
        epsilon=epsilon,
    )
    step_teacher_advantages = compute_teacher_token_advantage(
        response_mask=response_mask,
        teacher_log_prob=step_teacher_log_prob,
        old_log_prob=old_log_prob,
        critical_step_mask=step_skill_mask,
        normalize=normalize_teacher_adv,
        clip_range=clip_teacher_adv,
        epsilon=epsilon,
    )
    teacher_advantages = (
        episode_skill_weight * episode_teacher_advantages
        + step_skill_weight * step_teacher_advantages
    )

    scores = (
        episode_advantages
        + step_weight * step_advantages
        + teacher_advantages
    )
    step_group_metrics.update({
        "seed/adv/step_advantage_weight": step_weight,
        "seed/adv/episode_skill_teacher_weight": episode_skill_weight,
        "seed/adv/step_skill_teacher_weight": step_skill_weight,
    })
    return episode_advantages, step_advantages, teacher_advantages, scores, step_group_metrics

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   return_metrics: bool = False,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    remove_std = _mode_to_remove_std(mode)
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)
    step_group_metrics = compute_step_group_distribution_metrics(
        step_group_uids=step_group_uids,
        prefix="gigpo/state_group",
    )

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    if return_metrics:
        return scores, scores, step_group_metrics
    return scores, scores


def compute_seed_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   teacher_log_prob: Optional[torch.Tensor] = None,
                                   episode_teacher_log_prob: Optional[torch.Tensor] = None,
                                   step_teacher_log_prob: Optional[torch.Tensor] = None,
                                   old_log_prob: Optional[torch.Tensor] = None,
                                   critical_step_mask: Optional[torch.Tensor | np.ndarray | Sequence] = None,
                                   step_skill_mask: Optional[torch.Tensor | np.ndarray | Sequence] = None,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 0.0,
                                   episode_skill_teacher_advantage_w: float = 1.0,
                                   step_skill_teacher_advantage_w: float = 0.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   normalize_teacher_adv: bool = False,
                                   clip_teacher_adv: Optional[float] = None,
                                   return_metrics: bool = False,
                                   ):
    """
    Compute the advantages for SEED.

    SEED uses episode-level outcome advantages plus optional episode-skill and
    step-skill teacher terms, each derived from enhanced-prompt log-probs.
    """
    episode_advantages, step_advantages, teacher_advantages, scores, step_group_metrics = compute_seed_advantage_components(
        token_level_rewards=token_level_rewards,
        step_rewards=step_rewards,
        response_mask=response_mask,
        anchor_obs=anchor_obs,
        index=index,
        traj_index=traj_index,
        teacher_log_prob=teacher_log_prob,
        episode_teacher_log_prob=episode_teacher_log_prob,
        step_teacher_log_prob=step_teacher_log_prob,
        old_log_prob=old_log_prob,
        critical_step_mask=critical_step_mask,
        step_skill_mask=step_skill_mask,
        epsilon=epsilon,
        step_advantage_w=step_advantage_w,
        episode_skill_teacher_advantage_w=episode_skill_teacher_advantage_w,
        step_skill_teacher_advantage_w=step_skill_teacher_advantage_w,
        mode=mode,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
        normalize_teacher_adv=normalize_teacher_adv,
        clip_teacher_adv=clip_teacher_adv,
        metrics_prefix="seed/state_group",
    )
    if return_metrics:
        component_metrics = summarize_seed_advantage_components(
            response_mask=response_mask,
            episode_advantages=episode_advantages,
            step_advantages=step_advantages,
            teacher_advantages=teacher_advantages,
            step_advantage_w=step_advantage_w,
            epsilon=epsilon,
        )
        component_metrics.update(step_group_metrics)
        return scores, scores, component_metrics
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages
