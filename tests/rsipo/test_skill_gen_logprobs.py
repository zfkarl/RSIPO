"""_skill_gen_response_log_probs equals the answer-token log-probs of a full-sequence forward."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import verl.workers.actor.dp_actor as dp_actor
from verl.utils.torch_functional import logprobs_from_logits_v2


def _model():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


def _actor(model):
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    actor.actor_module = model
    actor.use_ulysses_sp = False
    actor.use_fused_kernels = False
    actor.use_remove_padding = True
    actor.device_name = "cpu"
    return actor


def _batch(samples, prompt_len, response_len):
    """Left-padded prompts followed by right-padded answers, as the skill-generation payload stores them."""
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


def _reference(model, ids, answer, temperature):
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(input_ids=ids, position_ids=torch.arange(ids.size(1)).unsqueeze(0), use_cache=False).logits[0]
    log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
    return log_probs[-answer - 1:-1].gather(-1, ids[0, -answer:].unsqueeze(-1)).squeeze(-1)


def test_response_log_probs_match_full_forward(monkeypatch):
    monkeypatch.setattr(dp_actor, "logprobs_from_logits", lambda logits, labels, inplace_backward=True: logprobs_from_logits_v2(logits, labels))
    model = _model()
    samples = [(9, 4), (12, 6), (5, 1)]
    batch = _batch(samples, prompt_len=12, response_len=6)
    got = _actor(model)._skill_gen_response_log_probs(batch, temperature=0.7)
    assert got.shape == (3, 6)
    for i, (p, a) in enumerate(samples):
        valid = batch["attention_mask"][i].bool()
        want = _reference(model, batch["input_ids"][i][valid].unsqueeze(0), a, 0.7)
        assert torch.allclose(got[i, :a], want, atol=3e-2), (i, got[i, :a], want)
        assert torch.all(got[i, a:] == 0)


def test_empty_answer_gives_zeros_but_keeps_the_graph(monkeypatch):
    monkeypatch.setattr(dp_actor, "logprobs_from_logits", lambda logits, labels, inplace_backward=True: logprobs_from_logits_v2(logits, labels))
    model = _model()
    batch = _batch([(7, 0)], prompt_len=8, response_len=4)
    got = _actor(model)._skill_gen_response_log_probs(batch, temperature=1.0)
    assert torch.all(got == 0) and got.requires_grad
    got.sum().backward()
