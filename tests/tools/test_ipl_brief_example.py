"""End-to-end tests for the IPL brief example (model replaced by a stub)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import openjarvis
from openjarvis.tools import cricket_brief as cb
from openjarvis.tools.cricket_facts import build_facts
from tests.tools.test_cricket_brief import _fixture_rows, _write

_PATH = Path(__file__).parents[2] / "examples" / "ipl_brief" / "ipl_brief.py"
_spec = importlib.util.spec_from_file_location("ipl_brief", _PATH)
ipl_brief = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ipl_brief)


def test_thinking_is_stripped():
    assert ipl_brief.strip_thinking("<think>hmm 999</think>\n## Brief") == "## Brief"


def _run(tmp_path, monkeypatch, replies, *extra):
    cb._CACHE.clear()
    seen = {"prompts": []}

    class FakeJarvis:
        def __init__(self, **kwargs):
            pass

        def ask(self, prompt, **kwargs):
            seen["prompts"].append(prompt)
            assert kwargs["context"] is False
            return replies[min(len(seen["prompts"]), len(replies)) - 1]

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(openjarvis, "Jarvis", FakeJarvis)
    out = tmp_path / "brief.md"
    monkeypatch.setattr(
        "sys.argv",
        ["ipl_brief", "--team", "MI", "--from-season", "2023",
         "--data-dir", str(tmp_path), "--out", str(out), *extra],
    )  # fmt: skip
    code = ipl_brief.main()
    return code, seen, out.read_text() if out.exists() else ""


def _season(tmp_path):
    from tests.tools.test_cricket_facts import _season_rows

    _write(tmp_path / "all_matches.csv", _season_rows())
    cb._CACHE.clear()
    report = cb.analyse(cb.load_deliveries(tmp_path), "Mumbai Indians")
    return build_facts(report)


def _run_season(tmp_path, monkeypatch, replies):
    import json as _json

    replies = [r if isinstance(r, str) else _json.dumps(r) for r in replies]
    return _run(tmp_path, monkeypatch, replies, "--from-season", "2024")


def test_bad_plan_then_good_plan(tmp_path, monkeypatch, capsys):
    from tests.tools.test_cricket_facts import _good_plan

    facts = _season(tmp_path)
    good = _good_plan(facts)
    bad = {**good, "attack": [{"fact": good["attack"][0]["fact"],
                               "note": "Harshal Patel is weak here"}]}  # fmt: skip
    code, seen, text = _run_season(tmp_path, monkeypatch, [bad, good])

    assert code == 0
    assert len(seen["prompts"]) == 2
    assert "names not on the fact sheet" in seen["prompts"][1]
    assert text.startswith("## Mumbai Indians: opposition brief")
    assert "Fact check: passed** (attempt 2)" in text
    assert "Rejected drafts (why)" in text and "Attempt 1:" in text
    assert "names not on the fact sheet" in capsys.readouterr().err
    assert seen["closed"]


def test_prose_reply_is_retried_then_falls_back(tmp_path, monkeypatch):
    _season(tmp_path)
    prose = "## MI brief\n- They struggle at the death"
    code, seen, text = _run_season(tmp_path, monkeypatch, [prose])

    assert code == 3
    assert len(seen["prompts"]) == ipl_brief.MAX_ATTEMPTS
    assert "not a single valid JSON object" in seen["prompts"][1]
    assert "opposition brief (fact sheet)" in text
    assert "model brief rejected after 3 attempts" in text
    assert "Last rejected model draft" in text


def test_facts_only_needs_no_model(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "all_matches.csv", _fixture_rows())
    code, seen, _ = _run(tmp_path, monkeypatch, ["unused"], "--facts-only")
    assert code == 0 and seen["prompts"] == []
    assert "FACTS (quote numbers only" in capsys.readouterr().out


def test_stats_only_needs_no_model(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "all_matches.csv", _fixture_rows())
    code, seen, _ = _run(tmp_path, monkeypatch, ["unused"], "--stats-only")
    assert code == 0 and seen["prompts"] == []
    out = capsys.readouterr().out
    assert "| powerplay | 15.00 |" in out and "F1 [" in out
