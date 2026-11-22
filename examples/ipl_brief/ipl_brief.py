#!/usr/bin/env python3
"""IPL opposition brief — fully offline scouting report from Cricsheet data.

Pipeline:
  1. ``cricket_brief`` computes every number in Python (no model involved).
  2. ``cricket_facts`` turns them into numbered facts with a verdict
     (STRENGTH / WEAKNESS / NEUTRAL) decided in Python.
  3. A local model (Ollama) returns a JSON plan: which fact IDs go in each
     section, plus a short tactical note for each. Notes may not contain
     numbers, stat/comparison words, or names not on the sheet.
  4. Python validates the plan (attack = WEAKNESS facts, watch = STRENGTH
     facts, no low-sample facts) and writes the brief itself, printing every
     figure from the fact text. The claim checker runs on the result as a
     backstop. Failures go back to the model (max 3 attempts); after that the
     brief falls back to the plain fact sheet. At worst plain, never wrong.

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

PLAN_PROMPT = """You are a T20 performance analyst for an IPL franchise,
planning against {team}. Below is a numbered FACTS sheet. Python will print
every number and name from the facts you choose; your job is only to choose
facts and add one short tactical note to each.

Reply with ONE JSON object and nothing else:
{{
  "bottom_line": {{"fact": "F?", "note": "one tactical sentence"}},
  "attack":  [{{"fact": "F?", "note": "how we exploit it"}}],
  "watch":   [{{"fact": "F?", "note": "how we handle it"}}],
  "players": [{{"fact": "F?", "note": "one-line plan for this player"}}]
}}

Rules (each is checked; breaking one means a retry):
- bottom_line: the single best WEAKNESS fact (not low sample).
- attack: 2-3 WEAKNESS facts. watch: 2-3 STRENGTH facts. players: 0-3 facts.
  Prefer the IDs listed under "Biggest relative weaknesses/strengths".
- Never use a low-sample fact in bottom_line, attack or watch.
- Notes are tactics only (field placings, lengths, match-ups, who bowls when),
  at most 25 words, with NO numbers and NO stat or comparison words
  (better, worse, higher, lower, league, average, economy, strike rate,
  runs, balls, wickets, strength, weak).
- In "watch" notes, describe how we contain them (e.g. "keep the boundary
  riders back"); avoid words like target, exploit, weak, struggle.
- Write player names exactly as on the sheet (e.g. "SA Yadav", not a full
  or different name). Mention no one who is not on the sheet.

{facts}
"""

RETRY_PROMPT = """Your plan was rejected. Problems:
{problems}

Reply again with ONE corrected JSON object only, following every rule.

{facts}
"""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


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
        parse_plan,
        render_fact_sheet,
        render_plan,
        validate_plan,
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

    brief, problems, attempts, draft = "", [], 0, ""
    history: list[tuple[int, list[str]]] = []
    prompt = PLAN_PROMPT.format(team=team, facts=sheet)
    try:
        for attempts in range(1, MAX_ATTEMPTS + 1):
            print(
                f"Planning brief for {team} with {args.model} "
                f"(attempt {attempts}/{MAX_ATTEMPTS})...",
                file=sys.stderr,
            )
            draft = strip_thinking(j.ask(prompt, context=False, temperature=0.2))
            plan = parse_plan(draft)
            problems = validate_plan(plan, facts)
            if not problems:
                brief = render_plan(team, plan, facts)
                # Backstop: the rendered brief must pass the claim checker too.
                problems = verify_brief(brief, facts, require_structure=True)
            if not problems:
                break
            history.append((attempts, problems))
            print(f"  check failed: {len(problems)} problem(s)", file=sys.stderr)
            for p in problems:
                print(f"    - {p}", file=sys.stderr)
            prompt = RETRY_PROMPT.format(
                problems="\n".join(f"- {p}" for p in problems), facts=sheet
            )
    finally:
        j.close()

    fell_back = bool(problems)
    rejected = draft
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
            f"\n**Fact check: passed** (attempt {attempts}). The model chose "
            "facts and wrote the tactical notes; every number, name and "
            "direction is printed from the cited fact."
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
