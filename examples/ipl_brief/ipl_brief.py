#!/usr/bin/env python3
"""IPL opposition brief — fully offline scouting report from Cricsheet data.

Pipeline:
  1. ``cricket_brief`` tool computes every number in Python (no model involved).
  2. A local model (Ollama) turns those numbers into a one-page brief.
  3. A number check flags any figure in the brief that is not in the stats.

Usage:
    python examples/ipl_brief/ipl_brief.py --team "Mumbai Indians"
    python examples/ipl_brief/ipl_brief.py --team KKR --from-season 2023
    python examples/ipl_brief/ipl_brief.py --team RCB --opponent CSK
    python examples/ipl_brief/ipl_brief.py --team MI --stats-only   # no model
    python examples/ipl_brief/ipl_brief.py --team MI --agent        # tool calling

Data: unzip Cricsheet's IPL "CSV (Ashwin)" download into
~/.openjarvis/data/cricket (or set OPENJARVIS_CRICKET_DATA).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

DEFAULT_MODEL = "qwen3.5:9b"

BRIEF_PROMPT = """You are a T20 performance analyst for an IPL franchise.
Write a one-page opposition brief on {team} for our coaches, using ONLY the
data summary below. Rules:
- Quote numbers exactly as they appear in the summary. Do not compute new ones.
- If a figure is marked "low sample", say it is a weak signal.
- No facts from outside the summary (no player form, injuries, auctions).

Format (markdown, under 350 words):
## {team}: opposition brief
**Bottom line:** one sentence on the single biggest opportunity.
### Where to attack their batting
3 bullets, each tied to a number.
### How to bat against their bowling
3 bullets, each tied to a number.
### Key players
2-3 bullets on named batters/bowlers.
### Confidence
One line on sample size and seasons covered.

DATA SUMMARY
{stats}
"""

AGENT_PROMPT = (
    "Use the cricket_brief tool to get stats for {team}{scope}, then write a "
    "one-page opposition brief for our coaches. Quote the tool's numbers "
    "exactly and do not invent any."
)

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def unsupported_numbers(brief: str, stats: str) -> list[str]:
    """Numbers in the brief that are not in the stats (in any rounding)."""
    allowed: set[str] = set()
    for raw in _NUM.findall(stats):
        exact = Decimal(raw)
        allowed.add(raw)
        for places in ("1", "0.1", "0.01"):  # half-up, as people round
            allowed.add(str(exact.quantize(Decimal(places), rounding=ROUND_HALF_UP)))
    flagged = []
    for raw in _NUM.findall(brief):
        value = float(raw)
        if raw in allowed:
            continue
        if value.is_integer() and (value <= 20 or 2000 <= value <= 2100):
            continue  # over numbers, counts like "top 3", seasons
        flagged.append(raw)
    return sorted(set(flagged), key=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="Team name or abbreviation")
    parser.add_argument("--from-season", type=int, default=None)
    parser.add_argument("--opponent", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engine", default="ollama")
    parser.add_argument("--data-dir", default=None, help="Cricsheet CSV folder")
    parser.add_argument(
        "--stats-only", action="store_true", help="Print computed stats, no model"
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Let the model call the tool itself (orchestrator agent)",
    )
    parser.add_argument("--out", default=None, help="Write the brief to this file")
    args = parser.parse_args()

    from openjarvis.tools.cricket_brief import CricketBriefTool

    params = {"team": args.team}
    if args.from_season:
        params["from_season"] = args.from_season
    if args.opponent:
        params["opponent"] = args.opponent

    stats = CricketBriefTool(data_dir=args.data_dir).execute(**params)
    if not stats.success:
        print(f"Error: {stats.content}", file=sys.stderr)
        return 1
    team = stats.metadata["team"]

    if args.stats_only:
        print(stats.content)
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

    print(f"Writing brief for {team} with {args.model} (local)...", file=sys.stderr)
    try:
        if args.agent:
            scope = f" against {args.opponent}" if args.opponent else ""
            if args.from_season:
                scope += f" from {args.from_season}"
            raw = j.ask(
                AGENT_PROMPT.format(team=team, scope=scope),
                agent="orchestrator",
                tools=["cricket_brief"],
                context=False,
            )
        else:
            raw = j.ask(
                BRIEF_PROMPT.format(team=team, stats=stats.content),
                context=False,
                temperature=0.2,
            )
    finally:
        j.close()

    brief = strip_thinking(raw)
    flagged = unsupported_numbers(brief, stats.content)

    footer = [
        "",
        "---",
        f"_Generated offline on {date.today()} by OpenJarvis ({args.model}) "
        "from Cricsheet ball-by-ball data. Numbers computed by the "
        "cricket_brief tool._",
    ]
    if flagged:
        footer.append(
            "\n**Number check: these figures are not in the computed stats, "
            f"verify before use:** {', '.join(flagged)}"
        )
    else:
        footer.append("\n**Number check: passed.** Every figure traces to the stats.")
    output = brief + "\n" + "\n".join(footer) + "\n"

    print(output)
    if args.out:
        Path(args.out).expanduser().write_text(output, encoding="utf-8")
        print(f"Saved to {args.out}", file=sys.stderr)
    return 2 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
