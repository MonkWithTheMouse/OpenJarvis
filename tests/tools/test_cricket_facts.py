"""Tests for the cricket fact sheet and claim checker.

The "gate 1" cases reproduce the six errors a qwen3.5:9b brief made on real
MI 2023-26 data, which the old "number appears anywhere" check let through.
"""

from __future__ import annotations

import pytest

from openjarvis.tools import cricket_brief as cb
from openjarvis.tools.cricket_facts import (
    NEUTRAL,
    STRENGTH,
    WEAKNESS,
    Fact,
    build_facts,
    fallback_brief,
    render_fact_sheet,
    verify_brief,
)
from tests.tools.test_cricket_brief import MI, _fixture_rows, _write


@pytest.fixture
def report(tmp_path):
    _write(tmp_path / "all_matches.csv", _fixture_rows())
    cb._CACHE.clear()
    return cb.analyse(cb.load_deliveries(tmp_path), MI, from_season=2023)


def _fact(fid, text, verdict, numbers):
    return Fact(id=fid, text=text, verdict=verdict, numbers=numbers)


# Facts shaped like the real MI 2023-26 output.
GATE1_FACTS = [
    _fact("F1", "Mumbai Indians middle batting run rate 9.07, above league 8.90",
          STRENGTH, ["9.07", "8.90"]),
    _fact("F2", "Tilak Varma (batter): 900 runs off 604 balls, strike rate 149, "
          "out 20 times", NEUTRAL, ["900", "604", "149", "20"]),
    _fact("F3", "Tilak Varma powerplay strike rate 126, lower than his overall 149",
          WEAKNESS, ["126", "149"]),
    _fact("F4", "Ryan Rickelton death strike rate 185, higher than his overall 168",
          STRENGTH, ["185", "168"]),
    _fact("F5", "Hardik Pandya (bowler): economy 10.55, 30 wickets in 700 balls",
          NEUTRAL, ["10.55", "30", "700"]),
    _fact("F6", "Deepak Chahar (bowler): economy 9.64, 25 wickets in 600 balls",
          NEUTRAL, ["9.64", "25", "600"]),
    _fact("F7", "Suryakumar Yadav (batter): 1500 runs off 900 balls, strike rate "
          "167, out 35 times", NEUTRAL, ["1500", "900", "167", "35"]),
    _fact("F8", "Mumbai Indians team-wide: 267 of 420 wickets lost were caught (64%)",
          NEUTRAL, ["267", "420", "64"]),
]  # fmt: skip


@pytest.mark.parametrize(
    "bad_line",
    [
        # 1. League figure presented as MI's, and negative wording on a strength.
        "**Bottom line:** MI score only 8.90 in the middle overs [F1]",
        # 2. Overall strike rate presented as a middle-overs figure.
        "- Tilak's strike rate jumps to 149 in the middle overs [F3]",
        # 3. A strength described as a struggle.
        "- Rickelton struggles at the death, strike rate 185 [F4]",
        # 4. Another player's number borrowed for comparison.
        "- Pandya's economy of 10.55 is worse than 9.64 [F5]",
        # 5. A player's own economy called the league figure.
        "- Chahar keeps it tight at 9.64, the league economy [F6]",
        # 6. Team-wide count pinned on one batter.
        "- Suryakumar has been out caught 267 times [F7]",
    ],
)
def test_gate1_errors_are_caught(bad_line):
    assert verify_brief(bad_line, GATE1_FACTS), bad_line


def test_correct_bullets_pass():
    brief = "\n".join(
        [
            "## Mumbai Indians: opposition brief",
            "**Bottom line:** Their middle-overs scoring of 9.07 beats the "
            "league's 8.90 [F1]",
            "### Where to attack them",
            "- Tilak Varma is slower in the powerplay: strike rate 126 vs 149 "
            "overall [F3]",
            "### What to watch out for",
            "- Rickelton accelerates at the death, 185 vs his usual 168 [F4]",
            "### Key players",
            "- Chahar: economy 9.64 with 25 wickets [F6]",
            "- 267 of 420 wickets were caught [F8]",
        ]
    )
    assert verify_brief(brief, GATE1_FACTS) == []


def test_uncited_numbers_and_unknown_ids():
    problems = verify_brief("- MI scored 9.07 an over\n- Fine line [F99]", GATE1_FACTS)
    assert any("Uncited" in p for p in problems)
    assert any("Unknown fact IDs" in p for p in problems)


def test_small_counts_and_seasons_do_not_need_citing():
    assert verify_brief("Seasons 2023-2026, top 3 threats, overs 16-20.", []) == []


def test_multi_cite_allows_numbers_from_both():
    line = "- Pandya 10.55 vs Chahar 9.64 [F5, F6]"
    assert verify_brief(line, GATE1_FACTS) == []


# ---------------------------------------------------------------------------
# Fact building on the hand-checked fixture
# ---------------------------------------------------------------------------


def test_fact_directions(report):
    facts = build_facts(report)
    by_text = {f.text.split(",")[0]: f for f in facts}
    pp = by_text["Mumbai Indians powerplay batting run rate 15.00"]
    # MI 15.00 vs league 12.00 -> above -> strength for MI.
    assert pp.verdict == STRENGTH and "above league 12.00" in pp.text
    assert pp.low_sample  # 2 balls
    death_bowl = by_text["Mumbai Indians death bowling economy 30.00"]
    # Bumrah 30.00 vs league death 16.50 (11 runs, 4 balls) -> weakness.
    assert death_bowl.verdict == WEAKNESS and "16.50" in death_bowl.text
    ids = [f.id for f in facts]
    assert ids == [f"F{i}" for i in range(1, len(facts) + 1)]


def test_fact_sheet_lists_every_fact(report):
    facts = build_facts(report)
    sheet = render_fact_sheet(facts)
    for f in facts:
        assert f.id in sheet
    assert "Biggest relative weaknesses" in sheet


def test_fallback_brief_always_passes_its_own_check(report):
    facts = build_facts(report)
    assert verify_brief(fallback_brief(MI, facts), facts) == []


def test_fallback_on_gate1_facts_passes():
    assert verify_brief(fallback_brief(MI, GATE1_FACTS), GATE1_FACTS) == []
