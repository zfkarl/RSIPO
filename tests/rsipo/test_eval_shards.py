from rsipo.eval_shards import shard_games


def test_one_shard_returns_every_game():
    assert shard_games(["a", "b", "c"], 1, 0) == ["a", "b", "c"]


def test_sixteen_shards_cover_both_splits_once():
    for total in (140, 134):
        games = list(range(total))
        parts = [shard_games(games, 16, index) for index in range(16)]
        assert sorted(game for part in parts for game in part) == games
        assert max(len(part) for part in parts) - min(len(part) for part in parts) <= 1
