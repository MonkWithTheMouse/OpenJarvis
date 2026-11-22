"""Tests for the IPL brief example's number check (no model needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[2] / "examples" / "ipl_brief" / "ipl_brief.py"
_spec = importlib.util.spec_from_file_location("ipl_brief", _PATH)
ipl_brief = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ipl_brief)

STATS = "| powerplay | 8.45 | League RR 8.91 | Balls/wkt 21.3 | Balls 2628 |"


def test_numbers_from_stats_pass():
    brief = "Powerplay run rate is 8.45 against a league 8.91, from 2628 balls."
    assert ipl_brief.unsupported_numbers(brief, STATS) == []


def test_rounded_numbers_pass():
    brief = "They score about 8.5 an over and lose a wicket every 21 balls."
    assert ipl_brief.unsupported_numbers(brief, STATS) == []


def test_invented_numbers_are_flagged():
    brief = "Their death strike rate of 142 is poor; economy 9.73 at the death."
    assert ipl_brief.unsupported_numbers(brief, STATS) == ["9.73", "142"]


def test_over_numbers_and_seasons_are_allowed():
    brief = "In overs 16-20 since 2023, attack them with 3 bowlers."
    assert ipl_brief.unsupported_numbers(brief, STATS) == []


def test_thinking_is_stripped():
    assert ipl_brief.strip_thinking("<think>hmm 999</think>\n## Brief") == "## Brief"


def test_end_to_end_with_fake_model(tmp_path, monkeypatch, capsys):
    """Full pipeline: real tool + fixture data, model replaced by a stub."""
    import openjarvis
    from openjarvis.tools import cricket_brief as cb
    from tests.tools.test_cricket_brief import _fixture_rows, _write

    _write(tmp_path / "all_matches.csv", _fixture_rows())
    cb._CACHE.clear()
    seen = {}

    class FakeJarvis:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def ask(self, prompt, **kwargs):
            seen["prompt"], seen["kwargs"] = prompt, kwargs
            return "<think>x</think>## Brief\nPowerplay run rate 15.00, death 99.9."

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(openjarvis, "Jarvis", FakeJarvis)
    out = tmp_path / "brief.md"
    monkeypatch.setattr(
        "sys.argv",
        ["ipl_brief", "--team", "MI", "--from-season", "2023",
         "--data-dir", str(tmp_path), "--out", str(out)],
    )  # fmt: skip
    code = ipl_brief.main()

    assert code == 2  # 99.9 is invented, so the check fails
    assert "| powerplay | 15.00 |" in seen["prompt"]
    assert seen["kwargs"]["context"] is False
    assert seen["closed"]
    text = out.read_text()
    assert "<think>" not in text
    assert "verify before use:** 99.9" in text
