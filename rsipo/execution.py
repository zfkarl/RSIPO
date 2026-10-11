"""Execution evidence shared by on-policy skill evolution and trust-calibrated self-distillation."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SKILL_VALUE = re.compile(r'"episode_skill"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
# ALFWorld game folders start with the task category; pick_and_place_simple must be tested last.
ALFWORLD_TASK_TYPES = (
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
    "pick_and_place",
)


def seed_for(*parts) -> int:
    raw = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def split_rollouts(seed: int, step: int, task_key: str, group_size: int, evidence_size: int) -> Tuple[List[int], List[int]]:
    """Assign rollouts to evidence/control before any outcome is inspected."""
    if not 0 < evidence_size < group_size:
        raise ValueError("evidence_size must leave a non-empty control set")
    order = list(range(group_size))
    random.Random(seed_for(seed, step, task_key, "split")).shuffle(order)
    return sorted(order[:evidence_size]), sorted(order[evidence_size:])


def choose_evidence(evidence_ids: Sequence[int], success: Dict[int, bool], seed: int, step: int, task_key: str) -> int:
    pool = [rollout for rollout in evidence_ids if success[rollout]] or list(evidence_ids)
    return random.Random(seed_for(seed, step, task_key, "evidence")).choice(pool)


def episode_utility(won: bool, steps: int, max_steps: int, efficiency_weight: float) -> float:
    """Failures score 0; a success scores 1 minus a length discount, so faster successes rank higher."""
    if not won:
        return 0.0
    return 1.0 - efficiency_weight * min(int(steps), int(max_steps)) / float(max_steps)


def execution_signals(
    delta: Optional[float],
    prior: float,
    teacher_rollouts: int,
    prior_strength: float,
    trust_kappa: float,
    trust_max: float,
) -> Tuple[float, float, float]:
    """Return (shrunk gain, OPD trust, skill reward); delta=None marks a task without valid evidence."""
    if delta is None:
        gain, reward = prior, 0.0
    else:
        gain = (teacher_rollouts * delta + prior_strength * prior) / (teacher_rollouts + prior_strength)
        reward = gain - prior
    trust = min(trust_max, max(0.0, 1.0 + trust_kappa * gain))
    return gain, trust, reward


@dataclass
class Reliability:
    """EMA of the mean execution gain of freshly generated skills."""

    eta: float = 0.1
    mean: float = 0.0
    completed_step: int = 0

    def commit(self, step: int, deltas: Sequence[float]) -> None:
        if step != self.completed_step + 1:
            raise ValueError(f"Expected reliability step {self.completed_step + 1}, got {step}")
        if deltas:
            self.mean = (1.0 - self.eta) * self.mean + self.eta * sum(deltas) / len(deltas)
        self.completed_step = int(step)

    def state(self) -> Dict[str, float]:
        return asdict(self)


def task_type(task_key: str) -> str:
    """Type-pooling group: ALFWorld's task category, Search's data source, and one group for WebShop."""
    key = str(task_key)
    if key.startswith("webshop:"):
        return "webshop"
    if key.startswith("{"):
        try:
            return "search:" + str(json.loads(key).get("d") or "unknown")
        except ValueError:
            return "search:unknown"
    for name in ALFWORLD_TASK_TYPES:
        if name in key:
            return name
    return "other"


def gain_noise_variance(teacher_mean: float, teacher_n: int, control_mean: float, control_n: int) -> float:
    """Sampling variance of delta = teacher mean utility - control mean utility.

    A utility in [0, 1] with mean mu has variance at most mu(1 - mu) (binomial when utility is plain success). The
    means are Laplace-smoothed, so an arm whose rollouts all failed (or all succeeded) still counts as noisy.
    """
    total = 0.0
    for mean, n in ((teacher_mean, teacher_n), (control_mean, control_n)):
        smoothed = (float(mean) * n + 1.0) / (n + 2.0)
        total += smoothed * (1.0 - smoothed) / n
    return total


@dataclass
class TypeStats:
    """Type pooling: per task type, EMAs over steps of the mean delta, mean delta^2 and mean noise variance.

    moments[type] = [EMA(mean delta), EMA(mean delta^2), EMA(mean v), EMA(1)]; dividing by the last entry removes the
    bias of the zero start. A type absent from a step is not decayed.
    """

    eta: float = 0.1
    moments: Dict[str, List[float]] = field(default_factory=dict)
    completed_step: int = 0

    def estimate(self, kind: str) -> Tuple[float, float]:
        """(type mean of delta, between-task variance); (0, 0) for a type not seen yet."""
        m = self.moments.get(kind)
        if not m or m[3] <= 0:
            return 0.0, 0.0
        mean = m[0] / m[3]
        spread = m[1] / m[3] - mean * mean
        return mean, max(0.0, spread - m[2] / m[3])

    def updated(self, step: int, observations: Dict[str, Sequence[Tuple[float, float]]]) -> "TypeStats":
        """A copy advanced by one step of (delta, noise variance) observations per type."""
        if step != self.completed_step + 1:
            raise ValueError(f"Expected type-statistics step {self.completed_step + 1}, got {step}")
        moments = {kind: list(values) for kind, values in self.moments.items()}
        for kind, pairs in observations.items():
            if not pairs:
                continue
            batch = (
                sum(d for d, _ in pairs) / len(pairs),
                sum(d * d for d, _ in pairs) / len(pairs),
                sum(v for _, v in pairs) / len(pairs),
                1.0,
            )
            m = moments.setdefault(kind, [0.0, 0.0, 0.0, 0.0])
            for index, value in enumerate(batch):
                m[index] = (1.0 - self.eta) * m[index] + self.eta * value
        return TypeStats(eta=self.eta, moments=moments, completed_step=int(step))

    def state(self) -> Dict[str, object]:
        return asdict(self)


def pooled_signals(
    delta: Optional[float],
    noise_var: float,
    type_mean: float,
    between_var: float,
    trust_kappa: float,
    trust_max: float,
) -> Tuple[float, float, float, float]:
    """Type pooling: (gain, OPD trust, skill reward, shrinkage weight) with the task's delta shrunk towards its type mean.

    weight = between_var / (between_var + noise_var): the noisier the task's own estimate relative to real differences
    between tasks of its type, the more its gain (and so its trust) is the type mean. The skill reward is the task's own
    delta against the type mean, not shrunk: shrinking it would zero every reward whenever the between-task spread is
    below the 4-vs-4 noise. delta=None marks a task without valid evidence.
    """
    if delta is None:
        gain, reward, weight = type_mean, 0.0, 0.0
    else:
        total = between_var + noise_var
        weight = between_var / total if total > 0 else 0.0
        gain = weight * delta + (1.0 - weight) * type_mean
        reward = delta - type_mean
    trust = min(trust_max, max(0.0, 1.0 + trust_kappa * gain))
    return gain, trust, reward, weight


def score_pooled(records, stats: TypeStats, step: int, trust_kappa: float, trust_max: float, format_penalty: float) -> TypeStats:
    """Fill gain/trust/skill_reward of one step's execution-check records with type pooling; returns the advanced statistics.

    The step's own deltas enter the type statistics before scoring, as in the method's algorithm; the caller keeps the
    returned statistics only once the actor update has succeeded.
    """
    observations: Dict[str, List[Tuple[float, float]]] = {}
    for record in records:
        record["task_type"] = task_type(record["game_file"])
        if record["delta"] is not None:
            record["noise_var"] = gain_noise_variance(
                record["teacher_utility"], len(record["teacher_success"]), record["control_utility"], len(record["control_ids"])
            )
            observations.setdefault(record["task_type"], []).append((record["delta"], record["noise_var"]))
    advanced = stats.updated(step, observations)
    for record in records:
        type_mean, between_var = advanced.estimate(record["task_type"])
        gain, trust, reward, weight = pooled_signals(
            record["delta"], record.get("noise_var", 0.0), type_mean, between_var, trust_kappa, trust_max
        )
        if not record["skill_valid"] and format_penalty > 0:
            reward = -format_penalty
        record.update(
            gain=gain, trust=trust, skill_reward=reward, shrink_weight=weight, type_mean=type_mean, type_between_var=between_var
        )
    return advanced


def skill_token_mask(tokenizer, token_ids: Sequence[int], text: str, skill: str) -> List[int]:
    """Mark response tokens that lie inside the JSON string value of episode_skill."""
    mask = [0] * len(token_ids)
    match = SKILL_VALUE.search(text)
    if match is None or not skill.strip():
        return mask
    try:
        if json.loads('"' + match.group(1) + '"').strip() != skill.strip():
            return mask
    except ValueError:
        return mask
    start, end = match.start(1), match.end(1)
    previous = 0
    for index in range(len(token_ids)):
        current = len(tokenizer.decode(list(token_ids[: index + 1]), skip_special_tokens=True))
        if current > end:
            break
        if previous >= start and current > previous:
            mask[index] = 1
        previous = current
    return mask


def format_token_mask(skill_mask: Sequence[int]) -> List[int]:
    """Response tokens around the episode_skill value: the JSON structure, the summary, the closing quote and brace, EOS."""
    return [1 - m for m in skill_mask] if any(skill_mask) else [0] * len(skill_mask)


def atomic_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
