"""SEED's RayPPOTrainer plus one execution check per step, feeding skill evolution and dynamic OPD trust."""
import json
import time
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask

from .execution import (
    Reliability,
    TypeStats,
    atomic_json,
    choose_evidence,
    episode_utility,
    execution_signals,
    format_token_mask,
    score_pooled,
    seed_for,
    skill_token_mask,
    split_rollouts,
)

METHOD_VERSION = "rsipo-v2"
GENERAL_SKILL_INSTRUCTION = (
    "Write the episode_skill as a general strategy that applies to similar questions: do not state the answer, "
    "and do not copy people's names, dates or text from the retrieved documents."
)


def _mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


class RsipoTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.rsipo
        actor = self.config.actor_rollout_ref.actor
        if self._is_seed_skill_gen_enabled():
            raise ValueError("Keep algorithm.seed.skill_gen.enable=False; RSIPO supplies the skill-generation reward")
        if not self._is_seed_policy_vllm_backend():
            raise ValueError("RSIPO regenerates skills through SEED's policy_vllm analysis backend")
        if float(actor.skill_gen_loss_coef) <= 0 or not bool(actor.skill_gen_same_step):
            raise ValueError("RSIPO trains skill generation in the same step: set actor.skill_gen_loss_coef>0 and actor.skill_gen_same_step=True")
        if float(actor.opd_loss_coef) <= 0:
            raise ValueError("RSIPO re-weights SEED's OPD loss, which is disabled")
        self.reliability = Reliability(eta=float(cfg.reliability_eta))
        # Gain, trust and skill reward come from the deltas pooled per task type across steps.
        self.type_stats = TypeStats(eta=float(cfg.reliability_eta))
        self._rsipo_analysis = None
        self._rsipo_pending = None
        self._rsipo_failures = 0

    def fit(self):
        if self.config.get("rsipo_eval") is not None:
            from .evaluation import evaluate

            return evaluate(self)
        return super().fit()

    def _collect_seed_skill_gen_samples(self, episode_analysis, analysis_tasks, metrics):
        self._rsipo_analysis = (episode_analysis, analysis_tasks)
        return super()._collect_seed_skill_gen_samples(episode_analysis, analysis_tasks, metrics)

    def _signals(self, delta, prior):
        cfg = self.config.rsipo
        return execution_signals(
            delta,
            prior,
            int(cfg.teacher_rollouts),
            float(cfg.prior_strength),
            float(cfg.trust_kappa),
            float(cfg.trust_max),
        )

    def before_actor_update(self, batch, metrics):
        cfg = self.config.rsipo
        step = int(self.global_steps)
        prior = float(self.reliability.mean)
        started = time.monotonic()
        error = None
        try:
            records, generated, skill_masks, format_masks, cost = self._execution_check(step)
            self._rsipo_failures = 0
        except Exception:
            self._rsipo_failures += 1
            if self._rsipo_failures > int(cfg.max_consecutive_failures):
                raise
            error = traceback.format_exc()
            print(f"RSIPO execution check failed at step {step}; this step falls back to the prior.\n{error}", flush=True)
            records, generated, skill_masks, format_masks, cost = [], None, None, None, {}

        type_stats = score_pooled(
            records, self.type_stats, step, float(cfg.trust_kappa), float(cfg.trust_max), float(cfg.format_penalty)
        )
        _, fallback_trust, _ = self._signals(None, prior)
        trust_by_task = {record["task"]: record["trust"] for record in records}

        sample_ids = np.asarray(batch.non_tensor_batch["sample_id"], dtype=np.int64)
        row_trust = torch.tensor([trust_by_task.get(int(s), fallback_trust) for s in sample_ids], dtype=torch.float32)
        batch.batch["rsipo_trust"] = row_trust
        metrics["rsipo/row_trust_mean"] = float(row_trust.mean())

        if generated is not None:
            batch.meta_info["seed_skill_gen"] = self._skill_payload(records, generated, skill_masks, format_masks)

        checked = [record for record in records if record["delta"] is not None]
        metrics.update(
            {
                "rsipo/execution_failed": float(error is not None),
                "rsipo/tasks": float(len(records)),
                "rsipo/skill_valid_rate": _mean(record["skill_valid"] for record in records),
                "rsipo/execution_valid_rate": len(checked) / len(records) if records else 0.0,
                "rsipo/initial_state_match_rate": _mean(
                    record["initial_state_matched"] for record in records if record["skill_valid"]
                ),
                "rsipo/teacher_success_rate": _mean(record["teacher_success_rate"] for record in checked),
                "rsipo/control_success_rate": _mean(record["control_success_rate"] for record in checked),
                "rsipo/teacher_utility": _mean(record["teacher_utility"] for record in checked),
                "rsipo/control_utility": _mean(record["control_utility"] for record in checked),
                "rsipo/success_delta_mean": _mean(record["success_delta"] for record in checked),
                "rsipo/format_penalized_frac": _mean(record.get("format_penalized", False) for record in records),
                "rsipo/delta_mean": _mean(record["delta"] for record in checked),
                "rsipo/delta_abs_mean": _mean(abs(record["delta"]) for record in checked),
                "rsipo/gain_mean": _mean(record["gain"] for record in records),
                "rsipo/reliability_before": prior,
                "rsipo/trust_mean": _mean(trust_by_task.values()) if records else fallback_trust,
                "rsipo/trust_min": min(trust_by_task.values()) if records else fallback_trust,
                "rsipo/trust_max": max(trust_by_task.values()) if records else fallback_trust,
                "rsipo/trust_below_one_frac": _mean(value < 1.0 for value in trust_by_task.values()),
                "rsipo/skill_reward_mean": _mean(record["skill_reward"] for record in records),
                "rsipo/skill_reward_abs_mean": _mean(abs(record["skill_reward"]) for record in records),
                "rsipo/skill_mask_tokens_mean": _mean(record["skill_mask_tokens"] for record in records),
                "rsipo/format_mask_tokens_mean": _mean(record.get("format_mask_tokens", 0) for record in records),
                "rsipo/evidence_success_rate": _mean(record["selected_evidence_success"] for record in records),
                "rsipo/seconds": time.monotonic() - started,
            }
        )
        metrics["rsipo/shrink_weight_mean"] = _mean(record["shrink_weight"] for record in checked)
        metrics["rsipo/noise_var_mean"] = _mean(record["noise_var"] for record in checked)
        for kind in sorted(type_stats.moments):
            type_mean, between_var = type_stats.estimate(kind)
            metrics[f"rsipo/type_mean/{kind}"] = type_mean
            metrics[f"rsipo/type_between_var/{kind}"] = between_var
        for key, value in cost.items():
            metrics["rsipo/cost/" + key] = float(value)
        self._rsipo_pending = {
            "step": step, "records": records, "error": error, "cost": cost, "prior": prior, "type_stats": type_stats,
        }
        return batch

    def after_actor_update(self, metrics):
        pending = self._rsipo_pending
        if pending is None or pending["step"] != self.global_steps:
            raise RuntimeError("RSIPO state does not belong to the current update")
        deltas = [record["delta"] for record in pending["records"] if record["delta"] is not None]
        self.reliability.commit(self.global_steps, deltas)
        metrics["rsipo/reliability_after"] = float(self.reliability.mean)
        type_stats_before = self.type_stats.state()
        self.type_stats = pending["type_stats"]
        run_root = Path(self.config.trainer.default_local_dir).parent
        atomic_json(
            run_root / "rsipo-audit" / f"step-{self.global_steps:05d}.json",
            {
                "step": self.global_steps,
                "method_version": METHOD_VERSION,
                "reliability_before": pending["prior"],
                "reliability_after": self.reliability.mean,
                "type_stats_before": type_stats_before,
                "type_stats_after": self.type_stats.state(),
                "error": pending["error"],
                "cost": pending["cost"],
                "records": pending["records"],
            },
        )
        self._rsipo_pending = None
        self._rsipo_analysis = None
        self.traj_collector.rsipo_context = None

    def _execution_check(self, step):
        cfg = self.config.rsipo
        context = getattr(self.traj_collector, "rsipo_context", None)
        if context is None or context["step"] != step:
            raise RuntimeError("RSIPO rollout context does not belong to this step")
        if self._rsipo_analysis is None:
            raise RuntimeError("SEED analysis was not captured for this step")
        episode_analysis, analysis_tasks = self._rsipo_analysis
        group = int(self.config.env.rollout.n)
        num_tasks = len(context["gen_batch"])

        trajectories, initial = {}, {}
        for traj, task, rollout, step_num, anchor in zip(
            context["traj_uid"], context["sample_id"], context["rollout_id"], context["step_num"], context["anchor_obs"]
        ):
            if trajectories.setdefault((int(task), int(rollout)), traj) != traj:
                raise RuntimeError("Rollout identity is not unique")
            if int(step_num) == 0:
                initial[traj] = str(anchor)
        lengths = Counter(context["traj_uid"])
        max_steps = int(self.config.env.max_steps)
        efficiency = float(cfg.efficiency_weight)

        records, uids_by_task = [], []
        for task in range(num_tasks):
            keys = context["task_keys"][task * group : (task + 1) * group]
            if len(set(keys)) != 1 or keys[0] in (None, "None"):
                raise RuntimeError(f"Rollout group {task} does not share one task")
            game = str(keys[0])
            uids = [trajectories[(task, rollout)] for rollout in range(group)]
            success = {
                rollout: float(analysis_tasks[uid].get("episode_success") or 0.0) >= 1.0 for rollout, uid in enumerate(uids)
            }
            evidence, control = split_rollouts(int(cfg.seed), step, game, group, int(cfg.evidence_rollouts))
            chosen = choose_evidence(evidence, success, int(cfg.seed), step, game)
            uids_by_task.append(uids)
            records.append(
                {
                    "task": task,
                    "game_file": game,
                    "evidence_ids": evidence,
                    "control_ids": control,
                    "selected_evidence_id": chosen,
                    "selected_evidence_success": int(success[chosen]),
                    "student_success": [int(success[rollout]) for rollout in range(group)],
                    "student_lengths": [int(lengths[uid]) for uid in uids],
                    "control_success_rate": _mean(success[rollout] for rollout in control),
                    "control_utility": _mean(
                        episode_utility(success[rollout], lengths[uids[rollout]], max_steps, efficiency) for rollout in control
                    ),
                    "initial_observation": initial.get(uids[control[0]]),
                    "delta": None,
                    "skill_valid": False,
                    "initial_state_matched": False,
                    "skill_mask_tokens": 0,
                }
            )

        analyzer = self._lazy_init_seed_analyzer()
        prompts, tasks = [], []
        for record, uids in zip(records, uids_by_task):
            uid = uids[record["selected_evidence_id"]]
            task_info = analysis_tasks[uid]
            prompt = (episode_analysis.get(uid) or {}).get("llm_prompt")
            if not prompt:
                prompt = analyzer._build_episode_analysis_prompt(
                    steps=task_info["steps"],
                    candidate_step_indices=task_info["candidate_step_indices"],
                    analysis_mode=task_info["analysis_mode"],
                    episode_success=task_info.get("episode_success"),
                    task_description=task_info.get("task_description"),
                )
            prompts.append(prompt)
            tasks.append(task_info)

        cost = {}
        session = self.traj_collector._start_rollout_generation_session(self.actor_rollout_wg)
        try:
            clock = time.monotonic()
            generated = self._generate_skills(prompts)
            cost["skill_generation_seconds"] = time.monotonic() - clock
            response_mask = compute_response_mask(generated).detach().cpu()
            responses = generated.batch["responses"].detach().cpu()
            skill_masks = torch.zeros(responses.shape, dtype=torch.float32)
            format_masks = torch.zeros(responses.shape, dtype=torch.float32)
            for index, (record, prompt, task_info) in enumerate(zip(records, prompts, tasks)):
                length = int(response_mask[index].sum())
                ids = responses[index][:length].tolist()
                content = self.tokenizer.decode(ids, skip_special_tokens=True)
                parsed = self._finalize_policy_vllm_seed_analysis(
                    analyzer,
                    content=content,
                    prompt=prompt,
                    steps=task_info["steps"],
                    candidate_step_indices=task_info["candidate_step_indices"],
                    analysis_mode=task_info["analysis_mode"],
                    task_description=task_info.get("task_description"),
                )
                skill = str(parsed.get("episode_skill") or "").strip()
                valid = not parsed.get("analysis_error") and analyzer._is_nontrivial_skill_text(skill)
                if valid:
                    token_mask = skill_token_mask(self.tokenizer, ids, content, skill)
                    skill_masks[index, :length] = torch.tensor(token_mask, dtype=torch.float32)
                    format_masks[index, :length] = torch.tensor(format_token_mask(token_mask), dtype=torch.float32)
                elif float(cfg.format_penalty) > 0:
                    # An unusable analysis is penalized as a whole, so format errors are also learned away.
                    skill_masks[index, :length] = 1.0
                    record["format_penalized"] = True
                record.update(
                    skill=skill,
                    raw_output=content,
                    skill_valid=bool(valid),
                    analysis_error=parsed.get("analysis_error"),
                    response_tokens=length,
                    skill_mask_tokens=int(skill_masks[index].sum()),
                    format_mask_tokens=int(format_masks[index].sum()),
                )
            cost["skill_generated_tokens"] = int(response_mask.sum())

            valid_records = [record for record in records if record["skill_valid"]]
            clock = time.monotonic()
            if valid_records:
                outcomes, teacher_lengths, starts, teacher_tokens = self.traj_collector.replay_with_skills(
                    context["gen_batch"],
                    self.actor_rollout_wg,
                    [record["game_file"] for record in valid_records],  # task key: game path, webshop id, or search spec
                    [record["skill"] for record in valid_records],
                    int(cfg.teacher_rollouts),
                    seed_for(int(cfg.seed), step, "replay"),
                )
                cost["teacher_generated_tokens"] = teacher_tokens
            cost["teacher_rollout_seconds"] = time.monotonic() - clock
        finally:
            self.traj_collector._stop_rollout_generation_session(self.actor_rollout_wg, session)

        count = int(cfg.teacher_rollouts)
        for position, record in enumerate(valid_records):
            won = outcomes[position * count : (position + 1) * count]
            steps_used = teacher_lengths[position * count : (position + 1) * count]
            reference = record["initial_observation"]
            starts_here = starts[position * count : (position + 1) * count]
            exact = reference is not None and all(start == reference for start in starts_here)
            relaxed = reference is not None and all(" ".join(start.split()) == " ".join(reference.split()) for start in starts_here)
            record["teacher_success"] = [int(value) for value in won]
            record["teacher_lengths"] = [int(value) for value in steps_used]
            record["teacher_success_rate"] = _mean(won)
            record["teacher_utility"] = _mean(
                episode_utility(value, used, max_steps, efficiency) for value, used in zip(won, steps_used)
            )
            record["success_delta"] = record["teacher_success_rate"] - record["control_success_rate"]
            record["initial_state_exact_match"] = bool(exact)
            record["initial_state_matched"] = bool(relaxed)
            if relaxed:
                record["delta"] = record["teacher_utility"] - record["control_utility"]
        cost["teacher_episodes"] = len(valid_records) * count
        return records, generated, skill_masks, format_masks, cost

    def _skill_prompt_text(self, prompt):
        text = self._seed_prompt_dict_to_text(prompt)
        if OmegaConf.select(self.config, "rsipo.skill_general_instruction", default=False):
            text = text.rstrip() + "\n\n" + GENERAL_SKILL_INSTRUCTION
        return text

    def _generate_skills(self, prompts):
        cfg = self.config.rsipo
        texts = [self._skill_prompt_text(prompt) for prompt in prompts]
        prompt_batch = self.traj_collector.build_prompt_batch(
            obs_contents=texts,
            data_sources=[None] * len(texts),
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": False,
                "sampling_params": {
                    "n": 1,
                    "temperature": float(cfg.skill_temperature),
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": int(self.config.algorithm.seed.analysis_max_completion_tokens),
                },
            },
            max_prompt_length=int(OmegaConf.select(self.config, "algorithm.seed.analysis_context_length") or 16384),
        )
        meta_info = deepcopy(prompt_batch.meta_info)
        request = prompt_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )
        request.meta_info = meta_info
        padded, pad_size = pad_dataproto_to_divisor(request, self.actor_rollout_wg.world_size)
        return unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)

    def _skill_payload(self, records, generated, skill_masks, format_masks):
        rewards = torch.tensor([record["skill_reward"] for record in records], dtype=torch.float32)
        payload = {key: generated.batch[key].detach().cpu() for key in ("input_ids", "attention_mask", "position_ids", "responses")}
        payload["rewards"] = rewards
        payload["loss_mask"] = skill_masks
        bonus = float(OmegaConf.select(self.config, "rsipo.format_bonus", default=0.0) or 0.0)
        if bonus > 0:
            payload["format_mask"] = format_masks
            payload["format_rewards"] = torch.tensor([bonus * record["skill_valid"] for record in records], dtype=torch.float32)
        rows = len(records)
        # Every rank must run the same number of FSDP forwards; padded rows carry no loss.
        extra = (-rows) % self.actor_rollout_wg.world_size
        if extra:
            index = torch.cat([torch.arange(rows), torch.zeros(extra, dtype=torch.long)])
            payload = {key: value.index_select(0, index) for key, value in payload.items()}
            payload["rewards"][rows:] = 0.0
            payload["loss_mask"][rows:] = 0.0
            if "format_mask" in payload:
                payload["format_rewards"][rows:] = 0.0
                payload["format_mask"][rows:] = 0.0
        payload["metrics"] = {
            "skill_gen/rows_with_skill_tokens": float((skill_masks.sum(-1) > 0).sum()),
            "skill_gen/padded_rows": float(extra),
        }
        return payload

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        atomic_json(
            folder / "rsipo-state.json",
            {
                "method_version": METHOD_VERSION,
                "reliability": self.reliability.state(),
                "type_stats": self.type_stats.state(),
                "rsipo": OmegaConf.to_container(self.config.rsipo, resolve=True),
            },
        )
        super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if self.global_steps == 0:
            return result
        if self.config.trainer.resume_mode == "resume_path":
            folder = Path(self.config.trainer.resume_from_path)
        else:
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        state = json.loads((folder / "rsipo-state.json").read_text())
        if state["rsipo"] != OmegaConf.to_container(self.config.rsipo, resolve=True):
            raise ValueError("Checkpoint RSIPO configuration differs from the current run")
        self.reliability = Reliability(**state["reliability"])
        if self.reliability.completed_step != self.global_steps:
            raise ValueError("RSIPO reliability state does not match the model checkpoint")
        self.type_stats = TypeStats(**state["type_stats"])
        if self.type_stats.completed_step != self.global_steps:
            raise ValueError("RSIPO type statistics do not match the model checkpoint")
        return result
