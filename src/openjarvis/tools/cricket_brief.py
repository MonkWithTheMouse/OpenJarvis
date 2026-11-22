"""Cricket opposition brief tool — offline T20 stats from Cricsheet CSV data.

Reads Cricsheet ball-by-ball data (JSON match files, or the "Ashwin" CSV
format) from a local folder and computes a deterministic scouting summary for one team:

* batting and bowling by phase (powerplay 1-6, middle 7-15, death 16-20),
  compared against the league average in the same data slice
* top batters and bowlers with phase splits
* dismissal mix
* auto-flagged weaknesses, with low-sample warnings

All numbers are computed in Python, never by the model, so the brief the agent
writes rests on figures that can be checked. The tool only reads ``.csv``/``.json``
files inside its data directory (default ``~/.openjarvis/data/cricket``) and
makes no network calls.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

TOOL_NAME = "cricket_brief"
DATA_DIR_ENV = "OPENJARVIS_CRICKET_DATA"
DEFAULT_DATA_DIR = Path.home() / ".openjarvis" / "data" / "cricket"

PHASES = ("powerplay", "middle", "death")
LOW_SAMPLE_BALLS = 60  # ten overs; below this a rate is flagged as noisy
VERDICT_MARGIN = 0.05  # >5% better/worse than the comparison = strength/weakness
FLAG_MARGIN = VERDICT_MARGIN  # one threshold for tables and fact verdicts

# Dismissals not credited to the bowler.
_NON_BOWLER_WICKETS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}
# Dismissals that do not cost the batting side a wicket.
_NOT_A_WICKET = {"retired hurt"}

# Franchise renames, so 2008-2025 data reads as one team.
_CANONICAL_TEAMS = {
    "delhi daredevils": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "rising pune supergiants": "Rising Pune Supergiant",
}
_ABBREVIATIONS = {
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "rcb": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "dc": "Delhi Capitals",
    "pbks": "Punjab Kings",
    "kxip": "Punjab Kings",
    "rr": "Rajasthan Royals",
    "gt": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
}

_REQUIRED_COLUMNS = {
    "match_id",
    "season",
    "innings",
    "ball",
    "batting_team",
    "bowling_team",
    "striker",
    "bowler",
    "runs_off_bat",
    "extras",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Delivery:
    match_id: str
    season_year: int
    innings: int
    over_number: int  # 1-based
    batting_team: str
    bowling_team: str
    striker: str
    bowler: str
    runs_off_bat: int
    extras: int
    wides: int
    noballs: int
    wicket_type: str
    player_dismissed: str

    @property
    def phase(self) -> str:
        if self.over_number <= 6:
            return "powerplay"
        if self.over_number <= 15:
            return "middle"
        return "death"

    @property
    def is_legal(self) -> bool:
        return self.wides == 0 and self.noballs == 0

    @property
    def faced_by_striker(self) -> bool:
        return self.wides == 0

    @property
    def team_runs(self) -> int:
        return self.runs_off_bat + self.extras

    @property
    def bowler_runs(self) -> int:
        # Byes and leg-byes are not charged to the bowler.
        return self.runs_off_bat + self.wides + self.noballs

    @property
    def is_team_wicket(self) -> bool:
        return bool(self.wicket_type) and self.wicket_type not in _NOT_A_WICKET

    @property
    def is_bowler_wicket(self) -> bool:
        return bool(self.wicket_type) and self.wicket_type not in _NON_BOWLER_WICKETS

    @property
    def is_boundary(self) -> bool:
        return self.runs_off_bat in (4, 6)


@dataclass
class Tally:
    runs: int = 0
    balls: int = 0
    wickets: int = 0
    dots: int = 0
    boundaries: int = 0

    def per_over(self) -> Optional[float]:
        return self.runs * 6 / self.balls if self.balls else None

    def strike_rate(self) -> Optional[float]:
        return self.runs * 100 / self.balls if self.balls else None

    def balls_per_wicket(self) -> Optional[float]:
        return self.balls / self.wickets if self.wickets else None

    def pct(self, part: int) -> Optional[float]:
        return part * 100 / self.balls if self.balls else None


@dataclass
class PlayerLine:
    name: str
    total: Tally = field(default_factory=Tally)
    by_phase: Dict[str, Tally] = field(
        default_factory=lambda: {p: Tally() for p in PHASES}
    )
    dismissals: Counter = field(default_factory=Counter)
    matches: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def canonical_team(name: str) -> str:
    cleaned = " ".join(name.split())
    return _CANONICAL_TEAMS.get(cleaned.lower(), cleaned)


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _season_year(value: str) -> int:
    digits = "".join(ch for ch in str(value)[:4] if ch.isdigit())
    return int(digits) if len(digits) == 4 else 0


def _over_number(ball: str) -> int:
    # Cricsheet writes "15.3" for the 3rd ball of the 16th over.
    head = str(ball).split(".", 1)[0]
    return _int(head) + 1


def _rows_to_deliveries(rows: Iterable[Dict[str, str]]) -> List[Delivery]:
    out: List[Delivery] = []
    for row in rows:
        innings = _int(row.get("innings"))
        if innings not in (1, 2):  # skip super overs
            continue
        out.append(
            Delivery(
                match_id=str(row.get("match_id", "")),
                season_year=_season_year(row.get("season", "")),
                innings=innings,
                over_number=_over_number(row.get("ball", "0")),
                batting_team=canonical_team(row.get("batting_team", "")),
                bowling_team=canonical_team(row.get("bowling_team", "")),
                striker=row.get("striker", "").strip(),
                bowler=row.get("bowler", "").strip(),
                runs_off_bat=_int(row.get("runs_off_bat")),
                extras=_int(row.get("extras")),
                wides=_int(row.get("wides")),
                noballs=_int(row.get("noballs")),
                wicket_type=(row.get("wicket_type") or "").strip().lower(),
                player_dismissed=(row.get("player_dismissed") or "").strip(),
            )
        )
    return out


def _json_match_rows(path: Path) -> List[Dict[str, str]]:
    """Flatten one Cricsheet JSON match into csv2-style row dicts."""
    try:
        match = json.loads(path.read_text(encoding="utf-8"))
        info = match["info"]
        innings_list = match["innings"]
    except (OSError, ValueError, KeyError, TypeError):
        return []  # not a Cricsheet match file (e.g. README export)
    teams = info.get("teams") or []
    rows: List[Dict[str, str]] = []
    for index, inn in enumerate(innings_list, start=1):
        if not isinstance(inn, dict):
            continue
        # Super overs are flagged; give them innings > 2 so they are skipped.
        number = index + 2 if inn.get("super_over") else index
        bat = inn.get("team", "")
        bowl = next((t for t in teams if t != bat), "")
        for over in inn.get("overs") or []:
            over_no = over.get("over", 0)
            for ball_no, d in enumerate(over.get("deliveries") or [], start=1):
                runs = d.get("runs") or {}
                extras = d.get("extras") or {}
                wickets = d.get("wickets") or [{}]
                rows.append(
                    {
                        "match_id": path.stem,
                        "season": str(info.get("season", "")),
                        "innings": str(number),
                        "ball": f"{over_no}.{ball_no}",
                        "batting_team": bat,
                        "bowling_team": bowl,
                        "striker": d.get("batter") or d.get("batsman", ""),
                        "bowler": d.get("bowler", ""),
                        "runs_off_bat": str(runs.get("batter", runs.get("batsman", 0))),
                        "extras": str(runs.get("extras", 0)),
                        "wides": str(extras.get("wides", "")),
                        "noballs": str(extras.get("noballs", "")),
                        "wicket_type": wickets[0].get("kind", ""),
                        "player_dismissed": wickets[0].get("player_out", ""),
                    }
                )
    return rows


def _csv_files(data_dir: Path) -> List[Path]:
    # Prefer the combined file if present; otherwise read per-match files.
    combined = data_dir / "all_matches.csv"
    if combined.is_file():
        return [combined]
    return sorted(
        p
        for p in data_dir.rglob("*.csv")
        if p.is_file() and not p.name.endswith("_info.csv")
    )


def _json_files(data_dir: Path) -> List[Path]:
    return sorted(p for p in data_dir.rglob("*.json") if p.is_file())


_CACHE: Dict[Tuple[str, Tuple[Tuple[str, float], ...]], List[Delivery]] = {}


def load_deliveries(data_dir: Path) -> List[Delivery]:
    files = _csv_files(data_dir) if data_dir.is_dir() else []
    if not files and data_dir.is_dir():
        files = _json_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No Cricsheet match files found in {data_dir}. Download the IPL "
            "JSON (or CSV) zip from cricsheet.org and unzip it there."
        )
    key = (str(data_dir), tuple((str(p), p.stat().st_mtime) for p in files))
    if key in _CACHE:
        return _CACHE[key]
    deliveries: List[Delivery] = []
    for path in files:
        if path.suffix == ".json":
            deliveries.extend(_rows_to_deliveries(_json_match_rows(path)))
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                # Not ball-by-ball csv2 data (e.g. a README export); skip it.
                continue
            deliveries.extend(_rows_to_deliveries(reader))
    if not deliveries:
        raise ValueError(
            f"Files in {data_dir} are not Cricsheet ball-by-ball data: need "
            "Cricsheet JSON match files or 'Ashwin' CSV "
            f"(columns: {', '.join(sorted(_REQUIRED_COLUMNS))})."
        )
    _CACHE.clear()
    _CACHE[key] = deliveries
    return deliveries


def resolve_team(query: str, known: Iterable[str]) -> Optional[str]:
    q = " ".join(query.split()).lower()
    if not q:
        return None
    if q in _ABBREVIATIONS:
        q = _ABBREVIATIONS[q].lower()
    q = canonical_team(q).lower()
    known = sorted(set(known))
    for team in known:
        if team.lower() == q:
            return team
    matches = [t for t in known if q in t.lower()]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _add(tally: Tally, d: Delivery, *, runs: int, ball_counts: bool) -> None:
    tally.runs += runs
    if ball_counts:
        tally.balls += 1
        if d.runs_off_bat == 0 and d.extras == 0:
            tally.dots += 1
    if d.is_boundary:
        tally.boundaries += 1


@dataclass
class TeamReport:
    team: str
    seasons: Tuple[int, int]
    matches: int
    batting: Dict[str, Tally]
    bowling: Dict[str, Tally]
    league_batting: Dict[str, Tally]
    league_bowling: Dict[str, Tally]
    batters: List[PlayerLine]
    bowlers: List[PlayerLine]
    dismissal_mix: Counter
    flags: List[str]


def analyse(
    deliveries: List[Delivery],
    team: str,
    *,
    from_season: Optional[int] = None,
    opponent: Optional[str] = None,
    top_n: int = 6,
) -> TeamReport:
    rows = [
        d for d in deliveries if from_season is None or d.season_year >= from_season
    ]
    if not rows:
        raise ValueError("No deliveries left after the season filter.")

    batting = {p: Tally() for p in PHASES}
    bowling = {p: Tally() for p in PHASES}
    league = {p: Tally() for p in PHASES}
    league_bowling = {p: Tally() for p in PHASES}
    batters: Dict[str, PlayerLine] = {}
    bowlers: Dict[str, PlayerLine] = {}
    dismissal_mix: Counter = Counter()
    matches: set = set()

    for d in rows:
        # League baseline: every team's batting in this slice.
        lt = league[d.phase]
        _add(lt, d, runs=d.team_runs, ball_counts=d.is_legal)
        if d.is_team_wicket:
            lt.wickets += 1
        _add(league_bowling[d.phase], d, runs=d.bowler_runs, ball_counts=d.is_legal)

        if opponent and opponent not in (d.batting_team, d.bowling_team):
            continue

        if d.batting_team == team:
            matches.add(d.match_id)
            t = batting[d.phase]
            _add(t, d, runs=d.team_runs, ball_counts=d.is_legal)
            if d.is_team_wicket:
                t.wickets += 1
                dismissal_mix[d.wicket_type] += 1
            line = batters.setdefault(d.striker, PlayerLine(d.striker))
            line.matches.add(d.match_id)
            for tally in (line.total, line.by_phase[d.phase]):
                _add(tally, d, runs=d.runs_off_bat, ball_counts=d.faced_by_striker)
            if d.is_team_wicket and d.player_dismissed == d.striker:
                line.total.wickets += 1
                line.by_phase[d.phase].wickets += 1
                line.dismissals[d.wicket_type] += 1
            elif d.is_team_wicket and d.player_dismissed in batters:
                other = batters[d.player_dismissed]  # e.g. non-striker run out
                other.total.wickets += 1
                other.by_phase[d.phase].wickets += 1
                other.dismissals[d.wicket_type] += 1

        elif d.bowling_team == team:
            matches.add(d.match_id)
            t = bowling[d.phase]
            _add(t, d, runs=d.bowler_runs, ball_counts=d.is_legal)
            if d.is_bowler_wicket:
                t.wickets += 1
            line = bowlers.setdefault(d.bowler, PlayerLine(d.bowler))
            line.matches.add(d.match_id)
            for tally in (line.total, line.by_phase[d.phase]):
                _add(tally, d, runs=d.bowler_runs, ball_counts=d.is_legal)
                if d.is_bowler_wicket:
                    tally.wickets += 1

    if not matches:
        raise ValueError(f"No deliveries for {team} in this data slice.")

    flags = _flags(team, batting, bowling, league, league_bowling, batters)
    years = sorted({d.season_year for d in rows if d.season_year})
    top_batters = sorted(batters.values(), key=lambda p: -p.total.runs)[:top_n]
    top_bowlers = sorted(bowlers.values(), key=lambda p: -p.total.balls)[:top_n]
    return TeamReport(
        team=team,
        seasons=(years[0], years[-1]) if years else (0, 0),
        matches=len(matches),
        batting=batting,
        bowling=bowling,
        league_batting=league,
        league_bowling=league_bowling,
        batters=top_batters,
        bowlers=top_bowlers,
        dismissal_mix=dismissal_mix,
        flags=flags,
    )


def _flags(
    team: str,
    batting: Dict[str, Tally],
    bowling: Dict[str, Tally],
    league: Dict[str, Tally],
    league_bowling: Dict[str, Tally],
    batters: Dict[str, PlayerLine],
) -> List[str]:
    flags: List[str] = []
    for phase in PHASES:
        ours, base = batting[phase], league[phase]
        rr, lrr = ours.per_over(), base.per_over()
        if rr is not None and lrr and rr < lrr * (1 - FLAG_MARGIN):
            flags.append(
                f"Batting {phase}: run rate {rr:.2f} vs league {lrr:.2f}"
                + _noise(ours.balls)
            )
        bpw, lbpw = ours.balls_per_wicket(), base.balls_per_wicket()
        if bpw is not None and lbpw and bpw < lbpw * (1 - FLAG_MARGIN):
            flags.append(
                f"Batting {phase}: loses a wicket every {bpw:.1f} balls "
                f"vs league {lbpw:.1f}" + _noise(ours.balls)
            )
        econ, lecon = bowling[phase].per_over(), league_bowling[phase].per_over()
        if econ is not None and lecon and econ > lecon * (1 + FLAG_MARGIN):
            flags.append(
                f"Bowling {phase}: economy {econ:.2f} vs league {lecon:.2f}"
                + _noise(bowling[phase].balls)
            )
    # Key batters who slow down at the death.
    for line in sorted(batters.values(), key=lambda p: -p.total.runs)[:6]:
        death = line.by_phase["death"]
        sr, dsr = line.total.strike_rate(), death.strike_rate()
        if sr and dsr is not None and death.balls >= 30 and dsr < sr:
            flags.append(
                f"{line.name}: death-overs strike rate {dsr:.0f} below "
                f"overall {sr:.0f} ({death.balls} balls)"
            )
    return flags


def _noise(balls: int) -> str:
    return f" (low sample: {balls} balls)" if balls < LOW_SAMPLE_BALLS else ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _f(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(report: TeamReport, opponent: Optional[str] = None) -> str:
    start, end = report.seasons
    scope = f"vs {opponent}" if opponent else "vs all opponents"
    lines = [
        f"# Data summary: {report.team} ({scope}, seasons {start}-{end}, "
        f"{report.matches} matches)",
        "",
        "## Batting by phase (team runs incl. extras)",
        "| Phase | Run rate | League RR | Balls/wkt | League balls/wkt | Balls |",
        "|---|---|---|---|---|---|",
    ]
    for p in PHASES:
        t, lg = report.batting[p], report.league_batting[p]
        lines.append(
            f"| {p} | {_f(t.per_over())} | {_f(lg.per_over())} | "
            f"{_f(t.balls_per_wicket(), 1)} | {_f(lg.balls_per_wicket(), 1)} | "
            f"{t.balls} |"
        )
    lines += [
        "",
        "## Bowling by phase (bowler-charged runs)",
        "| Phase | Economy | League econ | Wickets | Dot % | Balls |",
        "|---|---|---|---|---|---|",
    ]
    for p in PHASES:
        t = report.bowling[p]
        lines.append(
            f"| {p} | {_f(t.per_over())} | "
            f"{_f(report.league_bowling[p].per_over())} | {t.wickets} | "
            f"{_f(t.pct(t.dots), 1)} | {t.balls} |"
        )
    lines += [
        "",
        "## Top batters (by runs)",
        "| Batter | Runs | Balls | SR | Outs | PP SR | Death SR | Usual dismissal |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in report.batters:
        common = b.dismissals.most_common(1)
        lines.append(
            f"| {b.name} | {b.total.runs} | {b.total.balls} | "
            f"{_f(b.total.strike_rate(), 0)} | {b.total.wickets} | "
            f"{_f(b.by_phase['powerplay'].strike_rate(), 0)} | "
            f"{_f(b.by_phase['death'].strike_rate(), 0)} | "
            f"{common[0][0] if common else '-'} |"
        )
    lines += [
        "",
        "## Main bowlers (by balls bowled)",
        "| Bowler | Balls | Wkts | Economy | PP econ | Death econ |",
        "|---|---|---|---|---|---|",
    ]
    for b in report.bowlers:
        lines.append(
            f"| {b.name} | {b.total.balls} | {b.total.wickets} | "
            f"{_f(b.total.per_over())} | "
            f"{_f(b.by_phase['powerplay'].per_over())} | "
            f"{_f(b.by_phase['death'].per_over())} |"
        )
    total_wkts = sum(report.dismissal_mix.values())
    lines += ["", "## How their batters get out"]
    for kind, n in report.dismissal_mix.most_common():
        lines.append(f"- {kind}: {n} ({n * 100 / total_wkts:.0f}%)")
    lines += [
        "",
        f"## Auto-flagged weaknesses (>{FLAG_MARGIN:.0%} worse than league; "
        "computed, not model-generated)",
    ]
    lines += [f"- {f}" for f in report.flags] or [
        f"- None beyond {FLAG_MARGIN:.0%} of league."
    ]
    lines += [
        "",
        "Notes: phases are overs 1-6 / 7-15 / 16-20. League baseline = all "
        "teams in the same seasons. Figures under 60 balls are marked noisy.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR).expanduser()


@ToolRegistry.register("cricket_brief")
class CricketBriefTool(BaseTool):
    """Compute an offline scouting summary for a T20 team from Cricsheet CSVs."""

    tool_id = TOOL_NAME

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._data_dir = Path(data_dir).expanduser() if data_dir else None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=TOOL_NAME,
            description=(
                "Compute verified ball-by-ball stats for one IPL/T20 team from "
                "local Cricsheet data: batting and bowling by phase vs league "
                "average, top batters and bowlers, dismissal mix, and a FACTS "
                "list with IDs and verdicts. Use it before writing any "
                "opposition brief; cite fact IDs like [F3] and quote only "
                "numbers from the cited fact."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": (
                            "Team name or abbreviation, e.g. 'Mumbai Indians', "
                            "'CSK', 'KKR'."
                        ),
                    },
                    "from_season": {
                        "type": "integer",
                        "description": "Only use seasons from this year on, e.g. 2023.",
                    },
                    "opponent": {
                        "type": "string",
                        "description": "Optional: only matches against this team.",
                    },
                },
                "required": ["team"],
            },
            category="analytics",
            required_capabilities=["file:read"],
            timeout_seconds=60.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        data_dir = self._data_dir or _data_dir()
        try:
            deliveries = load_deliveries(data_dir)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return ToolResult(tool_name=TOOL_NAME, content=str(exc), success=False)

        known = {d.batting_team for d in deliveries}
        team = resolve_team(str(params.get("team", "")), known)
        if not team:
            return ToolResult(
                tool_name=TOOL_NAME,
                content=(
                    f"Unknown or ambiguous team {params.get('team')!r}. "
                    f"Known teams: {', '.join(sorted(known))}"
                ),
                success=False,
            )
        opponent = None
        if params.get("opponent"):
            opponent = resolve_team(str(params["opponent"]), known)
            if not opponent:
                return ToolResult(
                    tool_name=TOOL_NAME,
                    content=f"Unknown or ambiguous opponent {params['opponent']!r}.",
                    success=False,
                )
        from_season = params.get("from_season")
        try:
            from_season = int(from_season) if from_season not in (None, "") else None
            report = analyse(
                deliveries, team, from_season=from_season, opponent=opponent
            )
        except (TypeError, ValueError) as exc:
            return ToolResult(tool_name=TOOL_NAME, content=str(exc), success=False)
        from openjarvis.tools.cricket_facts import build_facts, render_fact_sheet

        content = render(report, opponent)
        content += "\n\n" + render_fact_sheet(build_facts(report))
        return ToolResult(
            tool_name=TOOL_NAME,
            content=content,
            success=True,
            metadata={"team": team, "matches": report.matches},
        )


__all__ = ["CricketBriefTool", "analyse", "load_deliveries", "render"]
