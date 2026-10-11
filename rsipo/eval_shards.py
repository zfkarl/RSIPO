"""Split an evaluation game list across servers without dropping or repeating a game."""


def shard_games(games, count, index):
    games = list(games)
    if count == 1 and index == 0:
        return games
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} of {count} is outside the list")
    return games[index::count]
