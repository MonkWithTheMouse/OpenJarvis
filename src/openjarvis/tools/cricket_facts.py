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

import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional

from openjarvis.tools.cricket_brief import (
    LOW_SAMPLE_BALLS,
    PHASES,
    VERDICT_MARGIN,
    Tally,
    TeamReport,
)

STRENGTH, WEAKNESS, NEUTRAL = "STRENGTH", "WEAKNESS", "NEUTRAL"
LEVEL_BAND = VERDICT_MARGIN  # within 5% of the comparison counts as level
MIN_PLAYER_BALLS = 30  # phase splits below this are too noisy to state

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_CITE = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
_NEGATIVE = re.compile(
    r"\b(weak\w*|struggl\w*|drop\w*|poor\w*|vulnerab\w*|slow\w*|leak\w*|"
    r"expensive|only|fail\w*|concern\w*|problem\w*|exploit\w*|target\w*|"
    r"worse)\b",
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
    r"tight|economical|impressive|best|better|strength\w*)\b",
    re.I,
)
# "after the powerplay" talks about a phase without claiming a figure for it.
_PHASE_QUALIFIER = re.compile(
    r"\b(after|before|outside|beyond|until|post|pre|beyond)\s+(the\s+)?$", re.I
)
_CAP_WORD = re.compile(r"\b[A-Z][A-Za-z'\-]*\b")
_ALWAYS_OK_CAPS = {
    "I", "PP", "IPL", "T20", "DRS", "Powerplay", "Middle", "Death", "Overs",
    "League", "Bottom", "Plan", "Why", "Biggest", "No", "Their", "They", "He",
    "His", "Key", "Where", "What",
}  # fmt: skip
_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")


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


REQUIRED_SECTIONS = ("### Where to attack them", "### What to watch out for")
BOTTOM_LINE = "**Bottom line:**"


def _structure_problems(brief: str) -> List[str]:
    problems: List[str] = []
    bottom = [ln for ln in brief.splitlines() if BOTTOM_LINE in ln]
    if not bottom:
        problems.append(f"Missing the '{BOTTOM_LINE}' line.")
    elif not _CITE.search(bottom[0]):
        problems.append("The bottom line must end with a fact ID like [F3].")
    for heading in REQUIRED_SECTIONS:
        if heading.lower() not in brief.lower():
            problems.append(f"Missing section heading '{heading}'.")
    return problems


def allowed_names(facts: List[Fact]) -> set:
    """Capitalised words that may appear: player and team names on the sheet."""
    names = set(_ALWAYS_OK_CAPS)
    for f in facts:
        head = re.split(
            r" \((?:batter|bowler)\)| (?:powerplay|middle|death) | "
            r"most often| team-wide",
            f.text,
        )[0]
        names.update(_CAP_WORD.findall(head))
    return names


def _unknown_names(text: str, names: set) -> List[str]:
    """Capitalised words that look like names but are not on the sheet.

    A sentence's first word may be capitalised. Any word directly before a
    known name word (e.g. "Sai" in "Sai Yadav") must itself be on the sheet,
    which catches renamed players even at the start of a sentence.
    """
    out = set()
    clean = re.sub(r"[*_`]", "", text)
    for sentence in re.split(r"(?<=[.;!?:])\s+", clean):
        words = list(_CAP_WORD.finditer(sentence))
        for i, m in enumerate(words):
            w = m.group(0)
            if w in names:
                continue
            first = i == 0 and not sentence[: m.start()].strip()
            nxt = words[i + 1] if i + 1 < len(words) else None
            before_name = (
                nxt is not None
                and not sentence[m.end() : nxt.start()].strip()
                and nxt.group(0) in names
                and nxt.group(0) not in _ALWAYS_OK_CAPS
            )
            if not first or before_name:
                out.add(w)
    return sorted(out)


def _phase_mentioned(pattern: re.Pattern, body: str) -> bool:
    for m in pattern.finditer(body):
        if not _PHASE_QUALIFIER.search(body[: m.start()]):
            return True
    return False


def _direction_problems(
    line: str, by_id: Dict[str, Fact], cites: List[str]
) -> List[str]:
    """Check wording per sentence, against the facts that sentence cites."""
    problems = []
    for sentence in _SENTENCE_SPLIT.split(line):
        own = [c.strip() for m in _CITE.findall(sentence) for c in m.split(",")]
        ids = [c for c in (own or cites) if c in by_id]
        if not ids:
            continue
        verdicts = {by_id[c].verdict for c in ids}
        body = _CITE.sub("", sentence)
        if _NEGATIVE.search(body) and STRENGTH in verdicts and WEAKNESS not in verdicts:
            problems.append(f"Negative wording on a STRENGTH fact: {line}")
        if _POSITIVE.search(body) and WEAKNESS in verdicts and STRENGTH not in verdicts:
            problems.append(f"Positive wording on a WEAKNESS fact: {line}")
    return problems


def verify_brief(
    brief: str, facts: List[Fact], *, require_structure: bool = False
) -> List[str]:
    """Return a list of problems; empty means every bullet is backed."""
    by_id: Dict[str, Fact] = {f.id: f for f in facts}
    names = allowed_names(facts)
    problems: List[str] = _structure_problems(brief) if require_structure else []
    for raw_line in brief.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cites = [c.strip() for m in _CITE.findall(line) for c in m.split(",")]
        body = _CITE.sub("", line)
        numbers = [n for n in _NUM.findall(body) if not _harmless(n)]
        strangers = _unknown_names(body, names)
        if strangers:
            problems.append(f"Names not on the fact sheet {strangers}: {line}")
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
            if _phase_mentioned(pattern, body) and phase not in cited_text:
                problems.append(
                    f"Mentions {phase} but cited {', '.join(cites)} "
                    f"is not about {phase}: {line}"
                )
        if _LEAGUE.search(body) and "league" not in cited_text:
            problems.append(
                f"Mentions league but cited {', '.join(cites)} has no league "
                f"figure: {line}"
            )
        problems += _direction_problems(line, by_id, cites)
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
    if weak:
        out.append(f"{BOTTOM_LINE} Biggest opening: {weak[0].text} [{weak[0].id}]")
    elif strong:
        out.append(
            f"{BOTTOM_LINE} No clear opening; their biggest edge: "
            f"{strong[0].text} [{strong[0].id}]"
        )
    else:
        out.append(f"{BOTTOM_LINE} No clear gaps beyond the margin [F1]")
    out += ["", REQUIRED_SECTIONS[0]]
    out += [f"- {f.text} [{f.id}]" for f in weak] or ["- No clear gaps."]
    out += ["", REQUIRED_SECTIONS[1]]
    out += [f"- {f.text} [{f.id}]" for f in strong] or ["- No clear gaps."]
    out += ["", "### Key players"]
    out += [f"- {f.text} [{f.id}]" for f in players[:4]]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Structured plan: the model picks facts, Python writes the brief
# ---------------------------------------------------------------------------

PLAN_SECTIONS = ("attack", "watch", "players")
MAX_NOTE_WORDS = 25
# Notes are tactics only. Anything that states or compares a figure is banned,
# because the figure is already printed from the fact itself.
_NOTE_BANNED = re.compile(
    r"\b(better|worse|higher|lower|above|below|strength\w*|weak\w*|strong\w*|"
    r"struggl\w*|league|average|economy|strike\s*rate|run\s*rate|percent|"
    r"times|wickets?|balls?|runs)\b",
    re.I,
)


def eligible(facts: List[Fact], verdict: str) -> List[Fact]:
    return [f for f in facts if f.verdict == verdict and not f.low_sample]


def parse_plan(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        plan = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return plan if isinstance(plan, dict) else None


def _note_problems(where: str, note: object, names: set) -> List[str]:
    if not isinstance(note, str) or not note.strip():
        return [f"{where}: 'note' must be a short non-empty string."]
    problems = []
    if len(note.split()) > MAX_NOTE_WORDS:
        problems.append(f"{where}: note is over {MAX_NOTE_WORDS} words.")
    if re.search(r"\d", note):
        problems.append(f"{where}: note must not contain numbers: {note!r}")
    banned = sorted({m.group(0).lower() for m in _NOTE_BANNED.finditer(note)})
    if banned:
        problems.append(
            f"{where}: note must be tactics only, no stat or comparison words "
            f"{banned}: {note!r}"
        )
    strangers = _unknown_names(note, names)
    if strangers:
        problems.append(
            f"{where}: names not on the fact sheet {strangers}; use names "
            f"exactly as written there: {note!r}"
        )
    return problems


def validate_plan(plan: Optional[dict], facts: List[Fact]) -> List[str]:
    """Problems with a model plan; empty means it can be rendered safely."""
    if plan is None:
        return ["Reply was not a single valid JSON object."]
    by_id = {f.id: f for f in facts}
    names = allowed_names(facts)
    weak_ok = {f.id for f in eligible(facts, WEAKNESS)}
    strong_ok = {f.id for f in eligible(facts, STRENGTH)}
    problems: List[str] = []

    def item(where: str, entry: object) -> Optional[Fact]:
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object with 'fact' and 'note'.")
            return None
        fid = str(entry.get("fact", "")).strip()
        problems.extend(_note_problems(where, entry.get("note"), names))
        if fid not in by_id:
            problems.append(f"{where}: unknown fact ID {fid!r}.")
            return None
        return by_id[fid]

    bottom = plan.get("bottom_line")
    fact = item("bottom_line", bottom)
    if fact is not None:
        allowed = weak_ok or strong_ok
        if allowed and fact.id not in allowed:
            kind = "WEAKNESS" if weak_ok else "STRENGTH"
            problems.append(
                f"bottom_line: {fact.id} is {fact.verdict}; pick a non-low-sample "
                f"{kind} fact."
            )

    rules = {"attack": (WEAKNESS, weak_ok), "watch": (STRENGTH, strong_ok)}
    for section in PLAN_SECTIONS:
        entries = plan.get(section, [])
        if not isinstance(entries, list):
            problems.append(f"{section}: must be a list.")
            continue
        if len(entries) > 3:
            problems.append(f"{section}: at most 3 items.")
        seen = set()
        for i, entry in enumerate(entries, 1):
            f = item(f"{section}[{i}]", entry)
            if f is None:
                continue
            if f.id in seen:
                problems.append(f"{section}[{i}]: {f.id} used twice.")
            seen.add(f.id)
            if section in rules:
                verdict, ok_ids = rules[section]
                if f.id not in ok_ids:
                    problems.append(
                        f"{section}[{i}]: {f.id} is {f.verdict}"
                        f"{' (low sample)' if f.low_sample else ''}; "
                        f"'{section}' needs a non-low-sample {verdict} fact."
                    )
        if section in rules and not entries and rules[section][1]:
            problems.append(f"{section}: give at least one item.")
    return problems


def _sentence(text: str) -> str:
    text = text.strip()
    text = text[0].upper() + text[1:] if text else text
    return text if text.endswith((".", "!", "?")) else text + "."


def render_plan(team: str, plan: dict, facts: List[Fact]) -> str:
    """Write the brief from a validated plan. Every figure is fact text."""
    by_id = {f.id: f for f in facts}

    def bullet(entry: dict) -> str:
        f = by_id[entry["fact"].strip()]
        return f"- {_sentence(f.text)} Plan: {_sentence(entry['note'])} [{f.id}]"

    b = plan["bottom_line"]
    bf = by_id[b["fact"].strip()]
    out = [
        f"## {team}: opposition brief",
        "",
        f"{BOTTOM_LINE} {_sentence(b['note'])} Why: {_sentence(bf.text)} [{bf.id}]",
        "",
        REQUIRED_SECTIONS[0],
    ]
    out += [bullet(e) for e in plan.get("attack", [])] or ["- No clear gaps."]
    out += ["", REQUIRED_SECTIONS[1]]
    out += [bullet(e) for e in plan.get("watch", [])] or ["- No clear gaps."]
    if plan.get("players"):
        out += ["", "### Key players"]
        out += [bullet(e) for e in plan["players"]]
    return "\n".join(out)


__all__ = [
    "Fact",
    "build_facts",
    "eligible",
    "fallback_brief",
    "parse_plan",
    "render_fact_sheet",
    "render_plan",
    "validate_plan",
    "verify_brief",
]
