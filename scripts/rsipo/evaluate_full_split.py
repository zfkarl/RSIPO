"""Run the public SEED evaluator with exactly one visit to every game in the selected split per round.

RSIPO_EVAL_SHARD_COUNT/INDEX optionally keeps every ``count``-th game (in sorted-path order) so several
processes cover the split once between them. Unset, the process still runs the whole split.
"""
import collections,hashlib,json,logging,os
from pathlib import Path
from examples.prompt_agent import local_vllm_alfworld as upstream
from scripts.sft._common.pipeline import build_manager
from agent_system.environments.env_package.alfworld.envs import load_config_file
from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from rsipo.eval_shards import shard_games


def full_coverage_env(args):
    full={'eval_in_distribution':140,'eval_out_of_distribution':134}[args.eval_dataset]
    config_path=upstream.PROJECT_ROOT/'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    config=load_config_file(str(config_path))
    assert config['env']['domain_randomization'] is False
    official=AlfredTWEnv(config,train_eval=args.eval_dataset)
    games=sorted(str(Path(x).resolve()) for x in official.game_files)
    assert len(games)==len(set(games))==full
    games=shard_games(games, int(os.environ.get('RSIPO_EVAL_SHARD_COUNT', '1')), int(os.environ.get('RSIPO_EVAL_SHARD_INDEX', '0')))
    expected=len(games)
    assert args.env_num==expected, f'shard has {expected} games, got --env-num {args.env_num}'
    manager=build_manager(game_files=games,alf_config_path=str(config_path),seed=args.seed,history_length=args.history_length)
    original_reset=manager.reset
    audit_path=Path(args.log_dir)/'coverage.json'
    manifest=[{'game_file':x,'sha256':hashlib.sha256(Path(x).read_bytes()).hexdigest(),'task_type':upstream.task_name_from_info({'extra.gamefile':x})} for x in games]
    assert all(x['task_type']!='other' for x in manifest)
    coverage={'expected_games':expected,'manifest':manifest,'category_counts':dict(collections.Counter(x['task_type'] for x in manifest)),'rounds':[]}
    def reset(kwargs):
        obs,infos=original_reset(kwargs)
        actual=[str(Path(x['extra.gamefile']).resolve()) for x in infos]
        assert len(actual)==len(set(actual))==expected,'Duplicate or missing evaluation game'
        assert set(actual)==set(games),'Evaluation split mismatch'
        coverage['rounds'].append({'round':len(coverage['rounds']),'unique_games':len(set(actual)),'duplicates':0,'missing':0,'actual_game_order':actual})
        audit_path.parent.mkdir(parents=True,exist_ok=True)
        audit_path.write_text(json.dumps(coverage,indent=2)+'\n')
        logging.info('Coverage verified: all %s games exactly once in %s', expected, args.eval_dataset)
        return obs,infos
    manager.reset=reset
    return manager

upstream.build_env=full_coverage_env
if __name__=='__main__':upstream.main()
