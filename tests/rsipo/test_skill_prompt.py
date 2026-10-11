"""rsipo.skill_general_instruction ends the skill-regeneration prompt with one instruction; off, the prompt is SEED's."""
from omegaconf import OmegaConf

from rsipo.trainer import GENERAL_SKILL_INSTRUCTION, RsipoTrainer

PROMPT = {"messages": [{"role": "user", "content": "Analyze ... return ONLY valid JSON.\n...\nAction valid: true\n\n"}]}
REMINDER = "Now return ONLY the JSON object with the two fields episode_summary and episode_skill."


def _trainer(rsipo):
    trainer = object.__new__(RsipoTrainer)
    trainer.config = OmegaConf.create({"algorithm": {"seed": {"analysis_prompt_reminder": True}}, "rsipo": rsipo})
    return trainer


def test_instruction_off_keeps_the_analysis_prompt():
    trainer = _trainer({})
    assert trainer._skill_prompt_text(PROMPT) == trainer._seed_prompt_dict_to_text(PROMPT)


def test_instruction_ends_the_prompt_after_the_reminder():
    text = _trainer({"skill_general_instruction": True})._skill_prompt_text(PROMPT)
    assert text.endswith("Action valid: true\n\n" + REMINDER + "\n\n" + GENERAL_SKILL_INSTRUCTION)
