# IPL Opposition Brief (fully offline)

Turns free Cricsheet ball-by-ball data into a one-page scouting brief on any
IPL team. Everything runs on your machine: no API keys, no cloud, no login.

```
Cricsheet CSVs ──► cricket_brief tool ──► local model (Ollama) ──► number check ──► brief.md
                   (Python computes        (writes prose only)      (flags any figure
                    every number)                                    not in the stats)
```

The design choice that matters: the model never does arithmetic. Phase run
rates, economies, strike rates and league baselines are computed in Python and
unit-tested by hand. The model only writes them up, and the number check
catches it if it makes a figure up (exit code 2).

## Setup on a Mac (Apple Silicon, 16 GB)

```bash
# 1. Code (your fork) + Python deps
git clone https://github.com/MonkWithTheMouse/OpenJarvis.git && cd OpenJarvis
git checkout <this branch>
uv sync                      # install uv first if needed: brew install uv

# 2. Local model engine
brew install ollama          # or download from https://ollama.com
ollama serve &               # leave running
ollama pull qwen3.5:9b       # ~6 GB; on 8 GB RAM use qwen3.5:4b

# 3. Data (free, Open Data Commons licence)
mkdir -p ~/.openjarvis/data/cricket
# From https://cricsheet.org/downloads/ download the Indian Premier League
# zip in "CSV (Ashwin)" format, then:
unzip ~/Downloads/ipl*_csv2.zip -d ~/.openjarvis/data/cricket
```

The tool reads `all_matches.csv` if the zip has one, otherwise every per-match
`<id>.csv` (it skips `<id>_info.csv`). Use `--data-dir` or the
`OPENJARVIS_CRICKET_DATA` variable to point somewhere else.

## Run

```bash
# 1. Check the numbers first (no model, instant)
uv run python examples/ipl_brief/ipl_brief.py --team MI --from-season 2023 --stats-only

# 2. Full brief
uv run python examples/ipl_brief/ipl_brief.py --team MI --from-season 2023 --out ~/mi_brief.md

# Head-to-head only
uv run python examples/ipl_brief/ipl_brief.py --team RCB --opponent CSK --from-season 2022

# Agentic mode: the model decides to call the tool itself
uv run python examples/ipl_brief/ipl_brief.py --team KKR --agent

# Or from the jarvis CLI
uv run jarvis ask --agent orchestrator --tools cricket_brief \
  "Write an opposition brief on Sunrisers Hyderabad since 2024"
```

Team names accept abbreviations (MI, CSK, RCB, KKR, SRH, DC, PBKS, RR, GT,
LSG). Old franchise names (Delhi Daredevils, Kings XI Punjab, Royal Challengers
Bangalore) are merged into the current ones.

## Testing it properly

1. `uv run pytest tests/tools/test_cricket_brief.py tests/tools/test_ipl_brief_example.py`
   checks the maths on a hand-checked fixture and the full pipeline with a stub model.
2. Spot-check a real number: run `--stats-only` for one team and season, then
   compare a top batter's runs and strike rate with ESPNcricinfo for that season.
3. Run the full brief 3 times. It should pass the number check every time. If
   it fails, the footer lists the invented figures.

## Limits (be honest about these in demos)

- No pace-vs-spin or venue splits: Cricsheet's ball-by-ball CSV has no bowler
  type. Adding a small bowler-type lookup file would be the next step.
- "Weakness" flags mean more than 8% worse than the league in the same seasons.
  They describe the past, not a prediction. Figures under 60 balls are marked
  as low sample.
- A 9B local model writes plainer prose than a frontier cloud model. The
  numbers are just as reliable, because the model doesn't produce them.
