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
    text = fallback_brief(MI, GATE1_FACTS)
    assert verify_brief(text, GATE1_FACTS, require_structure=True) == []


def test_structure_required_when_asked():
    facts = GATE1_FACTS
    bullets = "- Rickelton accelerates at the death, 185 vs his usual 168 [F4]"
    assert verify_brief(bullets, facts) == []
    problems = verify_brief(bullets, facts, require_structure=True)
    assert any("Bottom line" in p for p in problems)
    assert any("Where to attack them" in p for p in problems)
    assert any("What to watch out for" in p for p in problems)
    uncited = (
        "**Bottom line:** Attack early\n### Where to attack them\n"
        "### What to watch out for"
    )
    assert any(
        "must end with a fact ID" in p
        for p in verify_brief(uncited, facts, require_structure=True)
    )


def test_fallback_has_full_structure(report):
    facts = build_facts(report)
    text = fallback_brief(MI, facts)
    assert verify_brief(text, facts, require_structure=True) == []


def test_five_percent_band_is_neutral():
    from openjarvis.tools.cricket_facts import _verdict

    # Boult-like: 9.66 vs league 9.30 is ~3.9% worse -> NEUTRAL, not WEAKNESS.
    assert _verdict((9.30 - 9.66) / 9.30) == NEUTRAL
    assert _verdict(-0.06) == WEAKNESS and _verdict(0.06) == STRENGTH


# ---------------------------------------------------------------------------
# Probe cases from the a9fef50 run on real MI data (checker gaps)
# ---------------------------------------------------------------------------

PROBE_FACTS = [
    _fact("F6", "Mumbai Indians death batting loses a wicket every 11.3 balls, "
          "worse than league 12.1", WEAKNESS, ["11.3", "12.1"]),
    _fact("F10", "SA Yadav (batter): 1937 runs off 1150 balls, strike rate 168, "
          "out 48 times", NEUTRAL, ["1937", "1150", "168", "48"]),
    _fact("F12", "SA Yadav death strike rate 203, higher than his overall 168",
          STRENGTH, ["203", "168"]),
    _fact("F31", "JJ Bumrah (bowler): economy 7.17, 42 wickets in 889 balls",
          NEUTRAL, ["7.17", "42", "889"]),
    _fact("F37", "HH Pandya (bowler): economy 10.55, 29 wickets in 562 balls",
          NEUTRAL, ["10.55", "29", "562"]),
    _fact("F38", "HH Pandya powerplay economy 12.17, worse than league "
          "powerplay 9.30", WEAKNESS, ["12.17", "9.30"]),
    _fact("F46", "MJ Santner (bowler): economy 8.19, 15 wickets in 321 balls",
          NEUTRAL, ["8.19", "15", "321"]),
]  # fmt: skip


@pytest.mark.parametrize(
    "bad_line",
    [
        # Positive words on a weakness, cited alone.
        "- They lose a death wicket every 11.3 balls, significantly better "
        "than the league's 12.1 and their primary strength [F6]",
        # Mixed citations no longer skip the direction check.
        "- Pandya is impressive in the powerplay at 12.17 [F37, F38]",
        # Names not on the sheet.
        "- Harshal Patel goes at 12.17 in the powerplay [F38]",
        "- Sai Yadav scores at 203 at the death [F12]",
        "- Michael Santner keeps it to 8.19 [F46]",
    ],
)
def test_probe_gaps_are_caught(bad_line):
    assert verify_brief(bad_line, PROBE_FACTS), bad_line


def test_phase_qualifier_is_not_a_false_rejection():
    line = "- SA Yadav is dangerous after the powerplay; death strike rate 203 [F12]"
    assert verify_brief(line, PROBE_FACTS) == []


# ---------------------------------------------------------------------------
# Structured plan
# ---------------------------------------------------------------------------

_TEAMS = ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders",
          "Rajasthan Royals"]  # fmt: skip
_SURNAMES = ["Sharma", "Yadav", "Pandya", "Kumar", "Singh", "Iyer", "Rana"]


def _season_rows(seed=7, matches=60):
    import random

    from tests.tools.test_cricket_brief import _row

    r = random.Random(seed)
    rows = []
    for m in range(matches):
        a, b = r.sample(_TEAMS, 2)
        for inn, (bat, bowl) in enumerate([(a, b), (b, a)], 1):
            skill = 1.15 if bat == "Mumbai Indians" else 1.0
            for o in range(20):
                for ball in range(1, 7):
                    runs = r.choice([0, 0, 1, 1, 1, 2, 4, 6])
                    if r.random() < 0.1 * skill:
                        runs = r.choice([4, 6])
                    out = r.random() < (0.07 if o >= 15 else 0.04)
                    striker = f"{bat[:2].upper()} {_SURNAMES[r.randrange(5)]}"
                    bowler = f"{bowl[:2].upper()} {_SURNAMES[2 + r.randrange(5)]}"
                    rows.append(
                        _row(
                            m,
                            2024,
                            inn,
                            f"{o}.{ball}",
                            bat,
                            bowl,
                            striker,
                            bowler,
                            runs,
                            wicket="caught" if out else "",
                            out=striker if out else "",
                        )  # fmt: skip
                    )
    return rows


@pytest.fixture(scope="module")
def season_facts(tmp_path_factory):
    d = tmp_path_factory.mktemp("season")
    _write(d / "all_matches.csv", _season_rows())
    cb._CACHE.clear()
    report = cb.analyse(cb.load_deliveries(d), MI)
    return build_facts(report)


def _good_plan(facts):
    from openjarvis.tools.cricket_facts import eligible

    weak, strong = eligible(facts, WEAKNESS), eligible(facts, STRENGTH)
    assert weak and strong, "fixture must produce both verdicts"
    return {
        "bottom_line": {"fact": weak[0].id, "note": "go hard at them early"},
        "attack": [{"fact": f.id, "note": "set an attacking field"} for f in weak[:2]],
        "watch": [
            {"fact": f.id, "note": "keep the boundary riders back"} for f in strong[:2]
        ],  # fmt: skip
        "players": [],
    }


def test_good_plan_renders_and_passes_backstop(season_facts):
    from openjarvis.tools.cricket_facts import render_plan, validate_plan

    plan = _good_plan(season_facts)
    assert validate_plan(plan, season_facts) == []
    brief = render_plan(MI, plan, season_facts)
    assert brief.startswith("## Mumbai Indians: opposition brief")
    assert verify_brief(brief, season_facts, require_structure=True) == []
    # Figures come from the fact text verbatim.
    first = next(f for f in season_facts if f.id == plan["attack"][0]["fact"])
    assert first.text.split(",")[0] in brief


def _mutate(plan, section, idx=0, **changes):
    import copy

    p = copy.deepcopy(plan)
    target = p[section] if section == "bottom_line" else p[section][idx]
    target.update(changes)
    return p


@pytest.mark.parametrize(
    "mutation, expect",
    [
        (lambda p, f: _mutate(p, "attack", note="they score at 9.07"), "numbers"),
        (lambda p, f: _mutate(p, "attack", note="their league economy is poor"),
         "tactics only"),
        (lambda p, f: _mutate(p, "watch", note="Harshal Patel will open"),
         "names not on the fact sheet"),
        (lambda p, f: _mutate(p, "watch", fact=p["attack"][0]["fact"]), "needs"),
        (lambda p, f: _mutate(p, "attack", fact="F999"), "unknown fact ID"),
        (lambda p, f: {**p, "attack": []}, "at least one"),
    ],
)  # fmt: skip
def test_bad_plans_are_rejected(season_facts, mutation, expect):
    from openjarvis.tools.cricket_facts import validate_plan

    plan = mutation(_good_plan(season_facts), season_facts)
    problems = validate_plan(plan, season_facts)
    assert any(expect in p for p in problems), problems


def test_low_sample_fact_rejected_in_attack(season_facts):
    from dataclasses import replace

    from openjarvis.tools.cricket_facts import validate_plan

    plan = _good_plan(season_facts)
    fid = plan["attack"][0]["fact"]
    facts = [replace(f, low_sample=True) if f.id == fid else f for f in season_facts]
    assert any("low sample" in p for p in validate_plan(plan, facts))


def test_parse_plan_handles_prose_and_bad_json():
    from openjarvis.tools.cricket_facts import parse_plan, validate_plan

    assert parse_plan('Sure! {"bottom_line": {}} hope that helps') == {
        "bottom_line": {}
    }
    assert parse_plan("no json here") is None
    assert parse_plan("{broken") is None
    assert validate_plan(None, []) == ["Reply was not a single valid JSON object."]
