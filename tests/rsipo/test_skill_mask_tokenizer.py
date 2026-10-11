import json
import os
from pathlib import Path

import pytest

from rsipo.execution import skill_token_mask

MODEL = Path(os.environ.get("MODELS_ROOT", "/nonexistent")) / "Qwen2.5-3B-Instruct-alfworld-sft-glm52-ali"


@pytest.mark.skipif(not (MODEL / "tokenizer_config.json").exists(), reason="policy tokenizer not available")
def test_mask_with_policy_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    skill = 'If the target is "inside" a closed receptacle, open it first; heat it with microwave 1, then place it.'
    analysis = {"episode_summary": "The agent found the apple on countertop 1 and heated it.", "episode_skill": skill}
    text = "```json\n" + json.dumps(analysis, indent=2) + "\n```"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
    mask = skill_token_mask(tokenizer, ids, text, skill)
    covered = tokenizer.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=True)
    escaped = json.dumps(skill)[1:-1]
    assert covered in escaped
    assert len(covered) >= 0.9 * len(escaped)
