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


def _pp_fact(tmp_path):
    cb._CACHE.clear()
    rows = cb.load_deliveries(tmp_path)
    report = cb.analyse(rows, "Mumbai Indians", from_season=2023)
    return next(
        f
        for f in build_facts(report)
        if f.text.startswith("Mumbai Indians powerplay batting run rate")
    )


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


def test_bad_then_good_passes_on_retry(tmp_path, monkeypatch):
    _write(tmp_path / "all_matches.csv", _fixture_rows())
    fid = _pp_fact(tmp_path).id
    bad = f"<think>x</think>## Brief\n- MI struggle in the powerplay at 15.00 [{fid}]"
    good = f"## Brief\n- MI score quickly in the powerplay at 15.00 [{fid}]"
    code, seen, text = _run(tmp_path, monkeypatch, [bad, good])

    assert code == 0
    assert len(seen["prompts"]) == 2
    assert "Negative wording on a STRENGTH fact" in seen["prompts"][1]
    assert "Fact check: passed** (attempt 2)" in text
    assert "<think>" not in text and seen["closed"]


def test_always_bad_falls_back_to_fact_sheet(tmp_path, monkeypatch):
    _write(tmp_path / "all_matches.csv", _fixture_rows())
    fid = _pp_fact(tmp_path).id
    bad = f"## Brief\n- MI are weak in the powerplay at 15.00 [{fid}]"
    code, seen, text = _run(tmp_path, monkeypatch, [bad])

    assert code == 3
    assert len(seen["prompts"]) == ipl_brief.MAX_ATTEMPTS
    assert "opposition brief (fact sheet)" in text
    assert "model brief rejected after 3 attempts" in text
    assert "Rejected model draft" in text


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
