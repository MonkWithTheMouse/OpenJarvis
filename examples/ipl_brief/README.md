# IPL Opposition Brief (fully offline)

Turns free Cricsheet ball-by-ball data (JSON or CSV) into a one-page scouting brief on any
IPL team. Everything runs on your machine: no API keys, no cloud, no login.

```
Cricsheet data ─► cricket_brief ─► cricket_facts ─► local model ─► plan check ─► Python renders ─► claim check ─► brief.md
                  (every number)   (F1..Fn with      (JSON only:    (attack=WEAKNESS,  (numbers, names,   (backstop)
                                    STRENGTH /        fact IDs +     watch=STRENGTH,    direction copied
                                    WEAKNESS /        short tactic   no low sample,     from fact text)
                                    NEUTRAL)          notes)         notes: no numbers,
                                                                     stat words or
                                                                     unknown names)
            any failure ─► reasons fed back to the model (max 3 attempts) ─► then plain fact-sheet brief
```

The model never writes a number, a name or a direction. It returns JSON that
picks fact IDs for each section and adds a short tactical note to each
("keep the boundary riders back"). Python validates the plan and prints the
brief itself, copying every figure, player name and above/below verdict from
the fact text. Notes are rejected if they contain digits, stat or comparison
words (better, league, economy, strike rate...), or a name that is not on the
sheet. The rendered brief then goes through the claim checker as a backstop.
After three failed attempts you get the plain fact sheet: a plain brief,
never a wrong one.

How it got here (worth telling in a demo):

1. v1 checked only that each number appeared *somewhere* in the stats. On real
   MI 2023-26 data, qwen3.5:9b passed while pinning league figures on MI, team
   totals on one batter, and calling a 185 strike rate a struggle.
2. v2 added fact IDs and a rule-based claim checker. A probe then found it
   still missed "better"/"strength" on a weakness, skipped mixed citations,
   and let renamed players ("Harshal Patel" for HH Pandya) through.
3. v3 (this) stops patching word lists and removes the model's chance to get
   facts wrong: it only chooses and annotates. All errors from steps 1-2 are
   regression tests in `tests/tools/test_cricket_facts.py`.

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
   footer. Exit code 0 = model brief passed; 3 = fell back to the fact sheet.
   Every failed attempt's reasons are printed live and kept under "Rejected
   drafts (why)" in the file, so you can tell prompt problems from checker
   problems.

## Limits (be honest about these in demos)

- No pace-vs-spin or venue splits: Cricsheet's ball-by-ball CSV has no bowler
  type. Adding a small bowler-type lookup file would be the next step.
- STRENGTH / WEAKNESS means more than 5% better / worse than the comparison
  (league in the same seasons, or the player's own overall figure); within 5%
  is NEUTRAL. One constant, `VERDICT_MARGIN` in `cricket_brief.py`, drives both
  the stats-table flags and the fact verdicts.
  They describe the past, not a prediction. Figures under 60 balls are marked
  as low sample.
- Facts and figures are reliable by construction. The tactical notes are the
  model's opinion and are only checked for leaks (numbers, stat words,
  unknown names), not for cricket sense. A coach should still read them.
- Player names are Cricsheet's (e.g. "SA Yadav"), so the brief uses them too.
- A 9B local model writes plainer prose than a frontier cloud model. The
  numbers are just as reliable, because the model doesn't produce them.
