#!/usr/bin/env python3
"""IPL opposition brief — fully offline scouting report from Cricsheet data.

Pipeline:
  1. ``cricket_brief`` computes every number in Python (no model involved).
  2. ``cricket_facts`` turns them into numbered facts with a verdict
     (STRENGTH / WEAKNESS / NEUTRAL) decided in Python.
  3. A local model (Ollama) rewrites chosen facts as a brief, citing [F#].
  4. A claim checker rejects any bullet whose numbers are not in the fact it
     cites, or whose wording contradicts the fact's verdict. The model gets
     the errors and retries; if it still fails, the brief falls back to the
     plain fact sheet. You never get a wrong brief, at worst a plain one.

Usage:
    python examples/ipl_brief/ipl_brief.py --team "Mumbai Indians"
    python examples/ipl_brief/ipl_brief.py --team KKR --from-season 2023
    python examples/ipl_brief/ipl_brief.py --team RCB --opponent CSK
    python examples/ipl_brief/ipl_brief.py --team MI --stats-only   # no model
    python examples/ipl_brief/ipl_brief.py --team MI --facts-only   # no model

Exit codes: 0 model brief passed, 3 fell back to fact sheet, 1 error.

Data: unzip Cricsheet's IPL JSON (or CSV) download into
~/.openjarvis/data/cricket (or set OPENJARVIS_CRICKET_DATA).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

DEFAULT_MODEL = "qwen3.5:9b"
MAX_ATTEMPTS = 3

BRIEF_PROMPT = """You are a T20 performance analyst for an IPL franchise.
Write a short opposition brief on {team} for our coaches, using ONLY the
FACTS below.

Hard rules:
1. Every bullet ends with the fact ID(s) it uses, e.g. [F7] or [F2, F9].
2. Only quote numbers that appear in the fact you cite. Do not compute,
   combine or move numbers between players, phases or columns.
3. Respect each fact's verdict. A STRENGTH fact is a threat to us; never call
   it weak, slow, poor or a struggle. A WEAKNESS fact is an opportunity.
   NEUTRAL facts are context only.
4. Prefer facts listed under "Biggest relative weaknesses/strengths".
   If a fact is marked low sample, say it is a weak signal.
5. No outside knowledge (form, injuries, auctions, venues).

Format (markdown, under 300 words; these headings are checked):
**Bottom line:** one sentence, ending with its fact ID.
### Where to attack them
2-3 bullets from WEAKNESS facts.
### What to watch out for
2-3 bullets from STRENGTH facts.
### Key players
2-3 bullets.

{facts}
"""

RETRY_PROMPT = """Your brief failed the fact check. Problems:
{problems}

Rewrite the whole brief, fixing every problem. Same rules and format as
before: cite [F#] on every bullet, quote only numbers from the cited fact,
respect each verdict.

{facts}
"""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def with_title(brief: str, team: str) -> str:
    """Python owns the title: drop any model-written H1/H2, add ours."""
    body = [ln for ln in brief.splitlines() if not re.match(r"^#{1,2}\s", ln)]
    return f"## {team}: opposition brief\n\n" + "\n".join(body).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="Team name or abbreviation")
    parser.add_argument("--from-season", type=int, default=None)
    parser.add_argument("--opponent", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engine", default="ollama")
    parser.add_argument("--data-dir", default=None, help="Cricsheet data folder")
    parser.add_argument(
        "--stats-only", action="store_true", help="Print stats tables, no model"
    )
    parser.add_argument(
        "--facts-only", action="store_true", help="Print the fact sheet, no model"
    )
    parser.add_argument("--out", default=None, help="Write the brief to this file")
    args = parser.parse_args()

    from openjarvis.tools import cricket_brief as cb
    from openjarvis.tools.cricket_facts import (
        build_facts,
        fallback_brief,
        render_fact_sheet,
        verify_brief,
    )

    params = {"team": args.team}
    if args.from_season:
        params["from_season"] = args.from_season
    if args.opponent:
        params["opponent"] = args.opponent

    stats = cb.CricketBriefTool(data_dir=args.data_dir).execute(**params)
    if not stats.success:
        print(f"Error: {stats.content}", file=sys.stderr)
        return 1
    if args.stats_only:
        print(stats.content)
        return 0

    # Rebuild the report (data is cached) to get structured facts.
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else cb._data_dir()
    deliveries = cb.load_deliveries(data_dir)
    known = {d.batting_team for d in deliveries}
    team = cb.resolve_team(args.team, known)
    opponent = cb.resolve_team(args.opponent, known) if args.opponent else None
    report = cb.analyse(
        deliveries, team, from_season=args.from_season, opponent=opponent
    )
    facts = build_facts(report)
    sheet = render_fact_sheet(facts)

    if args.facts_only:
        print(sheet)
        return 0

    from openjarvis import Jarvis

    try:
        j = Jarvis(model=args.model, engine_key=args.engine)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Error: could not start the engine ({exc}).\n"
            f"Run:  ollama serve   and   ollama pull {args.model}",
            file=sys.stderr,
        )
        return 1

    brief, problems, attempts = "", [], 0
    history: list[tuple[int, list[str]]] = []
    prompt = BRIEF_PROMPT.format(team=team, facts=sheet)
    try:
        for attempts in range(1, MAX_ATTEMPTS + 1):
            print(
                f"Writing brief for {team} with {args.model} "
                f"(attempt {attempts}/{MAX_ATTEMPTS})...",
                file=sys.stderr,
            )
            draft = strip_thinking(j.ask(prompt, context=False, temperature=0.2))
            brief = with_title(draft, team)
            problems = verify_brief(brief, facts, require_structure=True)
            if not problems:
                break
            history.append((attempts, problems))
            print(f"  fact check failed: {len(problems)} problem(s)", file=sys.stderr)
            for p in problems:
                print(f"    - {p}", file=sys.stderr)
            prompt = RETRY_PROMPT.format(
                problems="\n".join(f"- {p}" for p in problems), facts=sheet
            )
    finally:
        j.close()

    fell_back = bool(problems)
    rejected = brief
    if fell_back:
        brief = fallback_brief(team, facts)

    footer = [
        "",
        "---",
        f"_Generated offline on {date.today()} by OpenJarvis ({args.model}). "
        "Data: Cricsheet (cricsheet.org). Every number comes from the "
        "cricket_brief tool; [F#] refers to the fact sheet below._",
    ]
    if fell_back:
        footer.append(
            f"\n**Fact check: model brief rejected after {attempts} attempts; "
            "showing the plain fact sheet instead.**"
        )
    else:
        footer.append(
            f"\n**Fact check: passed** (attempt {attempts}). Every bullet's "
            "numbers and direction match the fact it cites."
        )
    footer += ["", "<details><summary>Fact sheet</summary>", "", sheet, "</details>"]
    if history:
        footer += ["", "<details><summary>Rejected drafts (why)</summary>", ""]
        for n, probs in history:
            footer.append(f"Attempt {n}:")
            footer += [f"- {p}" for p in probs]
        footer.append("</details>")
    output = brief + "\n" + "\n".join(footer) + "\n"
    if fell_back:
        output += (
            "\n<details><summary>Last rejected model draft</summary>\n\n"
            + rejected
            + "\n</details>\n"
        )

    print(output)
    if args.out:
        Path(args.out).expanduser().write_text(output, encoding="utf-8")
        print(f"Saved to {args.out}", file=sys.stderr)
    return 3 if fell_back else 0


if __name__ == "__main__":
    sys.exit(main())
