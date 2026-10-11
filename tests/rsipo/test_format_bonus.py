"""rsipo.format_bonus: a valid analysis's tokens around the episode_skill value get a positive reward of their own."""
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import verl.workers.actor.dp_actor as dp_actor
from rsipo.execution import format_token_mask, skill_token_mask
from verl.utils.torch_functional import logprobs_from_logits_v2

TOKENIZER = Path(os.environ.get("MODELS_ROOT", "/nonexistent")) / "Qwen2.5-3B-Instruct"


def test_no_skill_tokens_means_no_format_tokens():
    assert format_token_mask([0, 0, 0]) == [0, 0, 0]
    assert format_token_mask([0, 1, 1, 0]) == [1, 0, 0, 1]


@pytest.mark.skipif(not (TOKENIZER / "tokenizer_config.json").exists(), reason="policy tokenizer not available")
def test_format_tokens_are_the_response_around_the_skill():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    skill = "Search for the missing attribute first, then answer with the entity it names."
    text = json.dumps({"episode_summary": "The agent searched twice and answered.", "episode_skill": skill}, indent=2)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    mask = skill_token_mask(tokenizer, ids, text, skill)
    around = format_token_mask(mask)
    assert all(a + b == 1 for a, b in zip(mask, around))
    assert around[-1] == 1 and around[-2] == 1 and "}" in tokenizer.decode([ids[-2]])
    kept = tokenizer.decode([i for i, m in zip(ids, around) if m], skip_special_tokens=True)
    assert "episode_summary" in kept and "missing attribute" not in kept


def _actor():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "eager"
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    actor.actor_module = Qwen2ForCausalLM(config).eval()
    actor.use_ulysses_sp = False
    actor.use_fused_kernels = False
    actor.use_remove_padding = True
    actor.device_name = "cpu"
    actor.config = {}
    return actor


def _batch(samples, prompt_len, response_len):
    input_ids = torch.zeros(len(samples), prompt_len + response_len, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    responses = torch.zeros(len(samples), response_len, dtype=torch.long)
    for i, (p, a) in enumerate(samples):
        prompt, answer = torch.randint(1, 97, (p,)), torch.randint(1, 97, (a,))
        input_ids[i, prompt_len - p:prompt_len] = prompt
        input_ids[i, prompt_len:prompt_len + a] = answer
        attention_mask[i, prompt_len - p:prompt_len + a] = 1
        responses[i, :a] = answer
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "responses": responses}


@pytest.mark.parametrize("bonus", [0.0, 0.05])
def test_format_bonus_adds_a_separately_averaged_term(monkeypatch, bonus):
    monkeypatch.setattr(dp_actor, "logprobs_from_logits", lambda logits, labels, inplace_backward=True: logprobs_from_logits_v2(logits, labels))
    monkeypatch.setattr(dp_actor, "get_torch_device", lambda: torch.cpu)
    actor = _actor()
    batch = _batch([(9, 5), (6, 4)], prompt_len=10, response_len=5)
    skill = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    around = torch.tensor([[1, 0, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.float32)
    payload = {**batch, "rewards": torch.tensor([0.3, 0.0]), "loss_mask": skill}
    if bonus:
        payload.update(format_mask=around, format_rewards=torch.tensor([bonus, 0.0]))
    metrics = actor.backward_skill_gen_loss(payload, temperature=1.0, loss_coef=1.0)
    got = [p.grad.clone() for p in actor.actor_module.parameters()]
    actor.actor_module.zero_grad()
    log_probs = actor._skill_gen_response_log_probs(batch, temperature=1.0)[0]
    mean = lambda m: (log_probs * m).sum() / m.sum()
    (-(0.3 * mean(skill[0]) + bonus * mean(around[0])) / 2).backward()
    for g, p in zip(got, actor.actor_module.parameters()):
        assert torch.allclose(g, p.grad, atol=1e-6, rtol=1e-4)
    assert metrics["actor/skill_gen_format_local_rows"] == (1.0 if bonus else 0.0)
