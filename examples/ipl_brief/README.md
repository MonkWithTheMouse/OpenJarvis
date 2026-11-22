# IPL Opposition Brief (fully offline)

Turns free Cricsheet ball-by-ball data (JSON or CSV) into a one-page scouting brief on any
IPL team. Everything runs on your machine: no API keys, no cloud, no login.

```
Cricsheet data ─► cricket_brief ─► cricket_facts ─► local model ─► claim checker ─► brief.md
                  (every number)   (F1..Fn, each     (rewrites      (per bullet: numbers
                                    STRENGTH /        facts, must    belong to cited fact,
                                    WEAKNESS /        cite [F#])     wording matches verdict,
                                    NEUTRAL)                         phase/league match)
                                                         ▲               │ fail: retry with
                                                         └───────────────┘ errors (max 3)
                                                     still failing ─► plain fact-sheet brief
```

The model never does arithmetic and never decides direction. Python computes
every figure and labels every comparison (above/below league, higher/lower
than the player's own average). The model only rephrases facts and must cite
their IDs. The checker rejects any bullet that quotes a number from a
different fact, calls a STRENGTH weak (or the reverse), or names a phase or
"league" that the cited fact is not about. After three failed attempts you get
the plain fact sheet: a plain brief, never a wrong one.

Why: a first version only checked that each number appeared *somewhere* in the
stats. On real MI 2023-26 data, qwen3.5:9b passed that check while pinning
league figures on MI, team totals on one batter, and calling a 185 strike rate
a struggle. Those six errors are now regression tests in
`tests/tools/test_cricket_facts.py`.

## Setup on a Mac (Apple Silicon, 16 GB)

```bash
# 1. Code (your fork) + Python deps
git clone https://github.com/MonkWithTheMouse/OpenJarvis.git && cd OpenJarvis
git checkout <this branch>
uv sync --extra dev          # dev extra includes pytest; install uv first: brew install uv

# 2. Local model engine
brew install ollama          # or download from https://ollama.com
ollama serve &               # leave running
ollama pull qwen3.5:9b       # ~6 GB; on 8 GB RAM use qwen3.5:4b

# 3. Data (free, Open Data Commons licence)
mkdir -p ~/.openjarvis/data/cricket
# From https://cricsheet.org/downloads/ download the Indian Premier League
# JSON zip (or the CSV "Ashwin" zip), then:
unzip ~/Downloads/ipl*.zip -d ~/.openjarvis/data/cricket
```

The tool reads CSV first if present (`all_matches.csv`, or per-match
`<id>.csv`, skipping `<id>_info.csv`), otherwise every Cricsheet JSON match
file. Super overs are excluded. Use `--data-dir` or the
`OPENJARVIS_CRICKET_DATA` variable to point somewhere else.

## Run

```bash
# 1. Check the numbers first (no model, instant)
uv run python examples/ipl_brief/ipl_brief.py --team MI --from-season 2023 --stats-only
uv run python examples/ipl_brief/ipl_brief.py --team MI --from-season 2023 --facts-only

# 2. Full brief
uv run python examples/ipl_brief/ipl_brief.py --team MI --from-season 2023 --out ~/mi_brief.md

# Head-to-head only
uv run python examples/ipl_brief/ipl_brief.py --team RCB --opponent CSK --from-season 2022

# Or let an agent call the tool itself (no claim checker on this path)
uv run jarvis ask --agent orchestrator --tools cricket_brief \
  "Write an opposition brief on Sunrisers Hyderabad since 2024"
```

Team names accept abbreviations (MI, CSK, RCB, KKR, SRH, DC, PBKS, RR, GT,
LSG). Old franchise names (Delhi Daredevils, Kings XI Punjab, Royal Challengers
Bangalore) are merged into the current ones.

## Testing it properly

1. `uv run pytest tests/tools/test_cricket_brief.py tests/tools/test_cricket_facts.py tests/tools/test_ipl_brief_example.py`
   checks the maths on a hand-checked fixture, the claim checker against real
   past errors, and the retry/fallback pipeline with a stub model.
2. Spot-check a real number: run `--stats-only` for one team and season, then
   compare a top batter's runs and strike rate with ESPNcricinfo for that season.
3. Run the full brief 3 times and read it against the fact sheet in its
   footer. Exit code 0 = model brief passed; 3 = fell back to the fact sheet
   (the rejected draft and reasons are at the bottom).

## Limits (be honest about these in demos)

- No pace-vs-spin or venue splits: Cricsheet's ball-by-ball CSV has no bowler
  type. Adding a small bowler-type lookup file would be the next step.
- "Weakness" flags mean more than 8% worse than the league in the same seasons.
  They describe the past, not a prediction. Figures under 60 balls are marked
  as low sample.
- The checker is rule-based. It catches wrong numbers, wrong direction words
  and wrong phase/league labels, but not every subtle misreading (e.g. a
  correct number with a misleading adjective it doesn't list). Read the brief
  against its cited facts before sharing.
- A 9B local model writes plainer prose than a frontier cloud model. The
  numbers are just as reliable, because the model doesn't produce them.
