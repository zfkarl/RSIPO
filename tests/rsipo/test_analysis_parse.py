"""A degenerate analysis that nests brackets thousands deep is an unusable analysis, not a crash."""
import pytest

from seed.analysis import SEEDEpisodeAnalyzer


def _analyzer():
    analyzer = object.__new__(SEEDEpisodeAnalyzer)
    analyzer.skill_mode = "episode_only"
    analyzer.include_episode_summary = True
    return analyzer


@pytest.mark.parametrize("text", ['{"episode_skill": ' + "[" * 100000, "{" + '"a": {' * 50000, "{" + "[" * 100000 + "}"])
def test_deeply_nested_output_is_a_parse_error(text):
    with pytest.raises(ValueError):
        _analyzer()._parse_analysis_response(text)


def test_object_missing_only_its_closing_brace_parses():
    text = ('{\n  "episode_summary": "The agent clicked \'Next >\' to browse more options.",\n'
            '  "episode_skill": "Given the failed trajectory, identify the core mistake and warning signs to avoid."')
    parsed = _analyzer()._parse_analysis_response(text)
    assert parsed["episode_skill"].startswith("Given the failed trajectory")


def test_unclosed_garbage_is_not_repaired():
    with pytest.raises(ValueError):
        _analyzer()._parse_analysis_response('{"episode_skill": "a b c d e f g', )


def test_valid_analysis_still_parses():
    parsed = _analyzer()._parse_analysis_response('{"episode_summary": "s", "episode_skill": "search first then click buy"}')
    assert parsed["episode_skill"] == "search first then click buy"
