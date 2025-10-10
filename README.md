# SCIM

## What the program does

The `scim.py` script simulates (or ingests) competency events across grades,
competencies, and sub-points, then produces:

* Monthly cumulative 3‑D voxel snapshots that highlight every
  grade/competency/sub-point combination satisfied to date.
* An animated GIF that stitches the monthly voxel images into a quick
  progression reel.
* A line chart showing the monthly average grade index per competency and the
  overall average.
* A CSV export of the underlying events for further analysis.

Axes follow the convention described in the prompt:

* **X** — Grades `G1` → `G9` (rendered back to front to ease depth perception)
* **Y** — Competencies: Tech Knowledge, Mentoring, Business, Growth
* **Z** — Sub-points `S1` → `S5`

The default simulation generates 50 entries across 12 months with an upward
trend in grade index and sub-point coverage.

## Running the script

Install dependencies (preferably inside a virtual environment). The optional
`imageio` package enables GIF creation; without it the script will still render
all PNG charts but skip the animation.

```bash
pip install -r requirements.txt
```

Generate the visualizations with deterministic output:

```bash
python scim.py --seed 7
```

Results are written to the `output/` directory:

* `voxels_YYYY_MM.png` — Monthly voxel snapshots
* `voxel_progress.gif` — Animated GIF of the snapshots
* `competency_trends.png` — Line chart of monthly average grades
* `entries.csv` — Raw events used for the charts

### Tweaking the simulation

The following CLI options control the synthetic data:

* `--entries` — Number of simulated records (default: 50)
* `--months` — Tracking window in months (default: 12)
* `--trend-strength` — How quickly grades trend upward (default: 0.35)
* `--seed` — Random seed for reproducibility

### Using your own CSV

Provide a CSV with columns `timestamp`, `competency`, `subpoint`, and either
`grade` (e.g. `G4`) or `grade_idx` (0-based index). Then run:

```bash
python scim.py --input my_entries.csv
```

All plots and exports will be regenerated from the supplied data.

## Implementation notes

* Grades are treated as indices `0..8` internally (`G1..G9` for display).
* Monthly averages use arithmetic mean. Switch to median/max by adjusting
  `plot_competency_trends`.
* Dense front-layer voxels can obscure back layers; consult the generated GIF
  for temporal context.
