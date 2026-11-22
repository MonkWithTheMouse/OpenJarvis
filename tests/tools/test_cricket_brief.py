"""Tests for the offline cricket_brief tool.

The fixture is small enough to check every expected number by hand.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

from openjarvis.core.registry import ToolRegistry
from openjarvis.security.capabilities import (
    DEFAULT_TOOL_CAPABILITIES,
    canonical_tool_capabilities,
)
from openjarvis.tools import cricket_brief as cb
from openjarvis.tools.cricket_brief import CricketBriefTool

COLUMNS = [
    "match_id", "season", "start_date", "venue", "innings", "ball",
    "batting_team", "bowling_team", "striker", "non_striker", "bowler",
    "runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes",
    "penalty", "wicket_type", "player_dismissed", "other_wicket_type",
    "other_player_dismissed",
]  # fmt: skip

MI, CSK = "Mumbai Indians", "Chennai Super Kings"


def _row(match, season, inn, ball, bat, bowl, striker, bowler, runs=0, *,
         wides=0, noballs=0, legbyes=0, wicket="", out=""):  # fmt: skip
    extras = wides + noballs + legbyes
    return {
        "match_id": match, "season": season, "start_date": "", "venue": "",
        "innings": inn, "ball": ball, "batting_team": bat,
        "bowling_team": bowl, "striker": striker, "non_striker": "X",
        "bowler": bowler, "runs_off_bat": runs, "extras": extras,
        "wides": wides or "", "noballs": noballs or "", "byes": "",
        "legbyes": legbyes or "", "penalty": "", "wicket_type": wicket,
        "player_dismissed": out, "other_wicket_type": "",
        "other_player_dismissed": "",
    }  # fmt: skip


def _fixture_rows():
    return [
        # Match 1, 2023: MI bat first.
        _row(1, 2023, 1, "0.1", MI, CSK, "Rohit", "Deepak", 4),
        _row(1, 2023, 1, "0.2", MI, CSK, "Rohit", "Deepak", wides=1),
        _row(
            1,
            2023,
            1,
            "0.2",
            MI,
            CSK,
            "Rohit",
            "Deepak",
            0,
            wicket="caught",
            out="Rohit",
        ),  # fmt: skip
        _row(1, 2023, 1, "8.1", MI, CSK, "SKY", "Jadeja", 2),
        _row(1, 2023, 1, "16.1", MI, CSK, "Hardik", "Pathirana", 6),
        _row(1, 2023, 1, "16.2", MI, CSK, "Hardik", "Pathirana", 0, legbyes=1),
        _row(
            1,
            2023,
            1,
            "16.3",
            MI,
            CSK,
            "Hardik",
            "Pathirana",
            0,
            wicket="run out",
            out="Hardik",
        ),  # fmt: skip
        # CSK chase.
        _row(1, 2023, 2, "0.1", CSK, MI, "Gaikwad", "Bumrah", 1),
        _row(
            1,
            2023,
            2,
            "17.1",
            CSK,
            MI,
            "Dhoni",
            "Bumrah",
            0,
            wicket="bowled",
            out="Dhoni",
        ),  # fmt: skip
        _row(1, 2023, 2, "17.2", CSK, MI, "Jadeja", "Bumrah", 4, noballs=1),
        # Super over: must be ignored.
        _row(1, 2023, 3, "0.1", MI, CSK, "Rohit", "Deepak", 6),
        # Match 2, older season, old franchise name.
        _row(2, "2020/21", 1, "0.1", MI, "Delhi Daredevils", "Rohit", "Nortje", 1),
    ]


def _write(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    _write(tmp_path / "all_matches.csv", _fixture_rows())
    return tmp_path


def _deliveries(data_dir: Path):
    cb._CACHE.clear()
    return cb.load_deliveries(data_dir)


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------


def test_super_overs_are_skipped(data_dir):
    rows = _deliveries(data_dir)
    assert len(rows) == 11
    assert all(d.innings in (1, 2) for d in rows)


def test_over_numbers_and_phases():
    assert cb._over_number("0.1") == 1
    assert cb._over_number("5.6") == 6
    assert cb._over_number("15.3") == 16
    d = cb._rows_to_deliveries([_row(1, 2023, 1, "15.3", MI, CSK, "A", "B", 1)])[0]
    assert d.phase == "death"


def test_season_years():
    assert cb._season_year("2007/08") == 2007
    assert cb._season_year("2023") == 2023
    assert cb._season_year("") == 0


def test_franchise_renames_are_merged(data_dir):
    rows = _deliveries(data_dir)
    assert "Delhi Capitals" in {d.bowling_team for d in rows}
    assert "Delhi Daredevils" not in {d.bowling_team for d in rows}


def test_resolve_team_abbreviations_and_substrings():
    known = {MI, CSK, "Punjab Kings", "Delhi Capitals", "Royal Challengers Bengaluru"}
    assert cb.resolve_team("csk", known) == CSK
    assert cb.resolve_team("mumbai", known) == MI
    assert cb.resolve_team("Royal Challengers Bangalore", known) == (
        "Royal Challengers Bengaluru"
    )
    assert cb.resolve_team("Delhi Daredevils", known) == "Delhi Capitals"
    assert cb.resolve_team("Kings", known) is None  # ambiguous
    assert cb.resolve_team("", known) is None


def test_per_match_files_and_info_files(tmp_path):
    rows = _fixture_rows()
    _write(tmp_path / "1.csv", [r for r in rows if r["match_id"] == 1])
    _write(tmp_path / "2.csv", [r for r in rows if r["match_id"] == 2])
    (tmp_path / "1_info.csv").write_text("version,1.0\ninfo,team,MI\n")
    assert len(_deliveries(tmp_path)) == 11


def test_missing_data_dir_message(tmp_path):
    cb._CACHE.clear()
    with pytest.raises(FileNotFoundError, match="cricsheet.org"):
        cb.load_deliveries(tmp_path / "nothing-here")


def test_wrong_format_is_rejected(tmp_path):
    (tmp_path / "all_matches.csv").write_text("a,b\n1,2\n")
    cb._CACHE.clear()
    with pytest.raises(ValueError, match="Ashwin"):
        cb.load_deliveries(tmp_path)


# ---------------------------------------------------------------------------
# Numbers (each checked by hand against the fixture)
# ---------------------------------------------------------------------------


def test_team_batting_by_phase(data_dir):
    report = cb.analyse(_deliveries(data_dir), MI, from_season=2023)
    pp, mid, death = (report.batting[p] for p in cb.PHASES)
    # 4 + wide(1) + 0 = 5 runs from 2 legal balls, 1 wicket.
    assert (pp.runs, pp.balls, pp.wickets) == (5, 2, 1)
    assert pp.per_over() == pytest.approx(15.0)
    assert (mid.runs, mid.balls) == (2, 1)
    # 6 + leg-bye(1) + 0 = 7 runs from 3 balls; run out costs a wicket.
    assert (death.runs, death.balls, death.wickets) == (7, 3, 1)
    assert report.matches == 1


def test_team_bowling_charges_only_bowler_runs(data_dir):
    report = cb.analyse(_deliveries(data_dir), CSK, from_season=2023)
    death = report.bowling["death"]
    # Pathirana: 6 + 0 + 0; the leg-bye is not his; the run out is not his.
    assert (death.runs, death.balls, death.wickets) == (6, 3, 0)
    assert death.per_over() == pytest.approx(12.0)

    mi = cb.analyse(_deliveries(data_dir), MI, from_season=2023)
    bumrah_death = mi.bowling["death"]
    # Bowled (0) + no-ball with 4 off the bat = 5 runs, 1 legal ball, 1 wicket.
    assert (bumrah_death.runs, bumrah_death.balls, bumrah_death.wickets) == (5, 1, 1)


def test_batter_lines(data_dir):
    report = cb.analyse(_deliveries(data_dir), MI, from_season=2023)
    lines = {b.name: b for b in report.batters}
    rohit = lines["Rohit"]
    # The wide is not a ball faced.
    assert (rohit.total.runs, rohit.total.balls, rohit.total.wickets) == (4, 2, 1)
    assert rohit.dismissals["caught"] == 1
    assert lines["Hardik"].dismissals["run out"] == 1
    assert report.batters[0].name == "Hardik"  # most runs (6)


def test_league_baseline_covers_all_teams(data_dir):
    report = cb.analyse(_deliveries(data_dir), MI, from_season=2023)
    lg = report.league_batting["powerplay"]
    # MI 5 runs / 2 balls + CSK 1 run / 1 ball.
    assert (lg.runs, lg.balls) == (6, 3)
    # Bowler-charged death runs league-wide: Pathirana 6 + Bumrah 5, 4 balls;
    # the leg-bye is excluded.
    lb = report.league_bowling["death"]
    assert (lb.runs, lb.balls) == (11, 4)


def test_season_filter(data_dir):
    rows = _deliveries(data_dir)
    assert cb.analyse(rows, MI).matches == 2
    assert cb.analyse(rows, MI, from_season=2023).matches == 1
    with pytest.raises(ValueError):
        cb.analyse(rows, MI, from_season=2030)


def test_opponent_filter(data_dir):
    rows = _deliveries(data_dir)
    report = cb.analyse(rows, MI, opponent="Delhi Capitals")
    assert report.matches == 1
    assert report.batting["powerplay"].runs == 1


def test_low_sample_flags_are_marked(data_dir):
    report = cb.analyse(_deliveries(data_dir), MI, from_season=2023)
    assert report.flags
    rate_flags = [f for f in report.flags if f.startswith(("Batting", "Bowling"))]
    assert rate_flags and all("low sample" in f for f in rate_flags)


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def test_tool_is_registered_with_file_read_only():
    module = importlib.reload(cb)
    assert ToolRegistry.get("cricket_brief") is module.CricketBriefTool
    assert module.CricketBriefTool().spec.required_capabilities == ["file:read"]
    assert DEFAULT_TOOL_CAPABILITIES["cricket_brief"] == ["file:read"]
    assert canonical_tool_capabilities(CricketBriefTool()) == ["file:read"]


def test_execute_success(data_dir):
    cb._CACHE.clear()
    result = CricketBriefTool(data_dir=str(data_dir)).execute(
        team="mi", from_season=2023
    )
    assert result.success, result.content
    assert "Mumbai Indians" in result.content
    assert "| powerplay | 15.00 |" in result.content
    assert result.metadata["matches"] == 1


def test_execute_reads_env_dir(data_dir, monkeypatch):
    cb._CACHE.clear()
    monkeypatch.setenv(cb.DATA_DIR_ENV, str(data_dir))
    assert CricketBriefTool().execute(team="CSK").success


def test_execute_unknown_team(data_dir):
    cb._CACHE.clear()
    result = CricketBriefTool(data_dir=str(data_dir)).execute(team="Barcelona")
    assert not result.success
    assert "Known teams" in result.content


def test_execute_bad_season(data_dir):
    cb._CACHE.clear()
    result = CricketBriefTool(data_dir=str(data_dir)).execute(
        team="MI", from_season="abc"
    )
    assert not result.success


def test_execute_missing_data(tmp_path):
    cb._CACHE.clear()
    result = CricketBriefTool(data_dir=str(tmp_path / "none")).execute(team="MI")
    assert not result.success
    assert "cricsheet.org" in result.content


@pytest.mark.parametrize("agent_type", ["simple", "native_react", "orchestrator"])
def test_agent_resolver_instantiates_cricket_brief(agent_type):
    from openjarvis.agents.tool_resolver import resolve_agent_tools

    ToolRegistry.register_value("cricket_brief", CricketBriefTool)
    resolved = resolve_agent_tools(
        {"agent_type": agent_type, "config": {"tools": ["cricket_brief"]}},
        engine=None,
        model="test-model",
    )
    try:
        assert set(resolved.by_name) == {"cricket_brief"}
        assert resolved.openai_specs[0]["function"]["name"] == "cricket_brief"
    finally:
        resolved.close()
