"""Fact sheet and claim checker for the cricket opposition brief.

The model must only restate facts, never derive them. Each fact carries:

* an ID (``F1``...) the brief must cite on every bullet
* the exact numbers it is allowed to quote
* a verdict from the scouted team's point of view (STRENGTH / WEAKNESS /
  NEUTRAL), already decided in Python

``verify_brief`` then checks every bullet: the numbers must belong to the
facts it cites, and its wording must not contradict their verdict (e.g.
"struggles" on a STRENGTH). It catches figures pinned on the wrong player,
phase or column, which a "does this number appear anywhere" check misses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional

from openjarvis.tools.cricket_brief import (
    LOW_SAMPLE_BALLS,
    PHASES,
    Tally,
    TeamReport,
)

STRENGTH, WEAKNESS, NEUTRAL = "STRENGTH", "WEAKNESS", "NEUTRAL"
LEVEL_BAND = 0.02  # within 2% of the comparison counts as level
MIN_PLAYER_BALLS = 30  # phase splits below this are too noisy to state

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_CITE = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
_NEGATIVE = re.compile(
    r"\b(weak\w*|struggl\w*|drop\w*|poor\w*|vulnerab\w*|slow\w*|leak\w*|"
    r"expensive|only|fail\w*|concern\w*|problem\w*|exploit\w*|target\w*)\b",
    re.I,
)
_PHASE_WORDS = {
    "powerplay": re.compile(r"\b(power\s?play|pp)\b", re.I),
    "middle": re.compile(r"\bmiddle\b", re.I),
    "death": re.compile(r"\bdeath\b", re.I),
}
_LEAGUE = re.compile(r"\bleague\b", re.I)
_POSITIVE = re.compile(
    r"\b(strong\w*|excel\w*|dominan\w*|elite|threat\w*|dangerous|"
    r"tight|economical|impressive|best)\b",
    re.I,
)


@dataclass
class Fact:
    id: str
    text: str
    verdict: str
    gap_pct: float = 0.0  # signed: + good for the scouted team
    low_sample: bool = False
    numbers: List[str] = field(default_factory=list)

    def line(self) -> str:
        noise = " (low sample)" if self.low_sample else ""
        return f"{self.id} [{self.verdict}{noise}] {self.text}"


def _num(value: Optional[float], digits: int) -> Optional[str]:
    return None if value is None else f"{value:.{digits}f}"


def _verdict(gap: float) -> str:
    if gap > LEVEL_BAND:
        return STRENGTH
    if gap < -LEVEL_BAND:
        return WEAKNESS
    return NEUTRAL


def _word(gap: float, up: str, down: str) -> str:
    if abs(gap) <= LEVEL_BAND:
        return "level with"
    return up if gap > 0 else down


class _Builder:
    def __init__(self) -> None:
        self.facts: List[Fact] = []

    def add(self, text: str, gap: float, balls: int, numbers: List[str]) -> None:
        self.facts.append(
            Fact(
                id=f"F{len(self.facts) + 1}",
                text=text,
                verdict=_verdict(gap),
                gap_pct=round(gap * 100, 1),
                low_sample=balls < LOW_SAMPLE_BALLS,
                numbers=[n for n in numbers if n is not None],
            )
        )


def build_facts(report: TeamReport) -> List[Fact]:
    """Every comparison the brief may make, with direction decided in Python."""
    b = _Builder()
    team = report.team

    for p in PHASES:
        ours, lg = report.batting[p], report.league_batting[p]
        rr, lrr = ours.per_over(), lg.per_over()
        if rr is not None and lrr:
            gap = (rr - lrr) / lrr
            b.add(
                f"{team} {p} batting run rate {rr:.2f}, "
                f"{_word(gap, 'above', 'below')} league {lrr:.2f} "
                f"({ours.balls} balls)",
                gap,
                ours.balls,
                [_num(rr, 2), _num(lrr, 2), str(ours.balls)],
            )
        bpw, lbpw = ours.balls_per_wicket(), lg.balls_per_wicket()
        if bpw is not None and lbpw:
            gap = (bpw - lbpw) / lbpw
            b.add(
                f"{team} {p} batting loses a wicket every {bpw:.1f} balls, "
                f"{_word(gap, 'better than', 'worse than')} league {lbpw:.1f}",
                gap,
                ours.balls,
                [_num(bpw, 1), _num(lbpw, 1)],
            )

    for p in PHASES:
        ours, lg = report.bowling[p], report.league_bowling[p]
        econ, lecon = ours.per_over(), lg.per_over()
        if econ is not None and lecon:
            gap = (lecon - econ) / lecon  # lower economy is good for them
            b.add(
                f"{team} {p} bowling economy {econ:.2f}, "
                f"{_word(gap, 'better than', 'worse than')} league {lecon:.2f} "
                f"({ours.wickets} wickets in {ours.balls} balls)",
                gap,
                ours.balls,
                [_num(econ, 2), _num(lecon, 2), str(ours.wickets), str(ours.balls)],
            )

    for bat in report.batters:
        sr = bat.total.strike_rate()
        if sr is None:
            continue
        b.add(
            f"{bat.name} (batter): {bat.total.runs} runs off {bat.total.balls} "
            f"balls, strike rate {sr:.0f}, out {bat.total.wickets} times",
            0.0,
            bat.total.balls,
            [str(bat.total.runs), str(bat.total.balls), _num(sr, 0),
             str(bat.total.wickets)],
        )  # fmt: skip
        for p in ("powerplay", "death"):
            t = bat.by_phase[p]
            psr = t.strike_rate()
            if psr is None or t.balls < MIN_PLAYER_BALLS:
                continue
            gap = (psr - sr) / sr
            b.add(
                f"{bat.name} {p} strike rate {psr:.0f}, "
                f"{_word(gap, 'higher than', 'lower than')} his overall {sr:.0f} "
                f"({t.balls} balls)",
                gap,
                t.balls,
                [_num(psr, 0), _num(sr, 0), str(t.balls)],
            )
        if bat.dismissals:
            kind, n = bat.dismissals.most_common(1)[0]
            b.add(
                f"{bat.name} most often out {kind}: {n} of {bat.total.wickets}",
                0.0,
                bat.total.balls,
                [str(n), str(bat.total.wickets)],
            )

    for bowl in report.bowlers:
        econ = bowl.total.per_over()
        if econ is None:
            continue
        b.add(
            f"{bowl.name} (bowler): economy {econ:.2f}, {bowl.total.wickets} "
            f"wickets in {bowl.total.balls} balls",
            0.0,
            bowl.total.balls,
            [_num(econ, 2), str(bowl.total.wickets), str(bowl.total.balls)],
        )
        for p in ("powerplay", "death"):
            t: Tally = bowl.by_phase[p]
            pe, le = t.per_over(), report.league_bowling[p].per_over()
            if pe is None or not le or t.balls < MIN_PLAYER_BALLS:
                continue
            gap = (le - pe) / le
            b.add(
                f"{bowl.name} {p} economy {pe:.2f}, "
                f"{_word(gap, 'better than', 'worse than')} league {p} {le:.2f} "
                f"({t.balls} balls)",
                gap,
                t.balls,
                [_num(pe, 2), _num(le, 2), str(t.balls)],
            )

    total = sum(report.dismissal_mix.values())
    for kind, n in report.dismissal_mix.most_common(2):
        b.add(
            f"{team} team-wide: {n} of {total} wickets lost were {kind} "
            f"({n * 100 / total:.0f}%)",
            0.0,
            total,
            [str(n), str(total), f"{n * 100 / total:.0f}"],
        )
    return b.facts


def render_fact_sheet(facts: List[Fact]) -> str:
    """Fact list plus the biggest gaps either way, for the model to work from."""
    ranked = sorted(
        (f for f in facts if f.verdict != NEUTRAL and not f.low_sample),
        key=lambda f: f.gap_pct,
    )
    weakest = [f.id for f in ranked[:3]]
    strongest = [f.id for f in reversed(ranked[-3:])]
    lines = ["FACTS (quote numbers only from the fact you cite)"]
    lines += [f.line() for f in facts]
    lines += [
        "",
        f"Biggest relative weaknesses: {', '.join(weakest) or 'none'}",
        f"Biggest relative strengths: {', '.join(strongest) or 'none'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def _variants(raw: str) -> set:
    exact = Decimal(raw)
    out = {raw}
    for places in ("1", "0.1", "0.01"):
        out.add(str(exact.quantize(Decimal(places), rounding=ROUND_HALF_UP)))
    return out


def _harmless(raw: str) -> bool:
    value = float(raw)
    return value.is_integer() and (value <= 20 or 2000 <= value <= 2100)


def verify_brief(brief: str, facts: List[Fact]) -> List[str]:
    """Return a list of problems; empty means every bullet is backed."""
    by_id: Dict[str, Fact] = {f.id: f for f in facts}
    problems: List[str] = []
    for raw_line in brief.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cites = [c.strip() for m in _CITE.findall(line) for c in m.split(",")]
        body = _CITE.sub("", line)
        numbers = [n for n in _NUM.findall(body) if not _harmless(n)]
        if not cites:
            if numbers:
                problems.append(f"Uncited numbers {numbers}: {line}")
            continue
        unknown = [c for c in cites if c not in by_id]
        if unknown:
            problems.append(f"Unknown fact IDs {unknown}: {line}")
            continue
        cited = [by_id[c] for c in cites]
        allowed = set()
        for f in cited:
            for n in f.numbers:
                allowed |= _variants(n)
        wrong = [n for n in numbers if n not in allowed]
        if wrong:
            problems.append(
                f"Numbers {wrong} are not in cited {', '.join(cites)}: {line}"
            )
        cited_text = " ".join(f.text for f in cited).lower()
        for phase, pattern in _PHASE_WORDS.items():
            if pattern.search(body) and phase not in cited_text:
                problems.append(
                    f"Mentions {phase} but cited {', '.join(cites)} "
                    f"is not about {phase}: {line}"
                )
        if _LEAGUE.search(body) and "league" not in cited_text:
            problems.append(
                f"Mentions league but cited {', '.join(cites)} has no league "
                f"figure: {line}"
            )
        verdicts = {f.verdict for f in cited}
        if _NEGATIVE.search(body) and verdicts == {STRENGTH}:
            problems.append(f"Negative wording on a STRENGTH fact: {line}")
        if _POSITIVE.search(body) and verdicts == {WEAKNESS}:
            problems.append(f"Positive wording on a WEAKNESS fact: {line}")
    return problems


def fallback_brief(team: str, facts: List[Fact]) -> str:
    """Model-free brief built straight from the facts. Plain, never wrong."""
    ranked = sorted(
        (f for f in facts if f.verdict != NEUTRAL and not f.low_sample),
        key=lambda f: f.gap_pct,
    )
    weak = [f for f in ranked if f.verdict == WEAKNESS][:3]
    strong = [f for f in reversed(ranked) if f.verdict == STRENGTH][:3]
    players = [f for f in facts if "(batter)" in f.text or "(bowler)" in f.text]
    out = [f"## {team}: opposition brief (fact sheet)", ""]
    out.append("### Where they are weaker than the league")
    out += [f"- {f.text} [{f.id}]" for f in weak] or ["- No clear gaps."]
    out += ["", "### Where they are stronger than the league"]
    out += [f"- {f.text} [{f.id}]" for f in strong] or ["- No clear gaps."]
    out += ["", "### Key players"]
    out += [f"- {f.text} [{f.id}]" for f in players[:4]]
    return "\n".join(out)


__all__ = [
    "Fact",
    "build_facts",
    "fallback_brief",
    "render_fact_sheet",
    "verify_brief",
]
