"""algorithm.seed.analysis_prompt_reminder restates the JSON instruction after the trajectory; off, SEED's text is kept."""
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

PROMPT = {"messages": [{"role": "user", "content": "Analyze ... return ONLY valid JSON.\n...\nAction valid: true\n\n"}]}


def _trainer(config):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(config)
    return trainer


def test_reminder_off_keeps_seeds_prompt():
    assert _trainer({"algorithm": {"seed": {}}})._seed_prompt_dict_to_text(PROMPT) == PROMPT["messages"][0]["content"]


def test_reminder_follows_the_trajectory():
    text = _trainer({"algorithm": {"seed": {"analysis_prompt_reminder": True}}})._seed_prompt_dict_to_text(PROMPT)
    assert text.startswith("Analyze ... return ONLY valid JSON.")
    assert text.endswith("Action valid: true\n\nNow return ONLY the JSON object with the two fields episode_summary and episode_skill.")
