# SCIM

What the program does

Axes:

X = Grades G1–G9 (back→front)

Y = Competencies = Tech Knowledge, Mentoring, Business, Growth

Z = Sub-points S1–S5 (≥5 satisfied)

Simulation: 50 entries, timestamps spread over 12 months, with an upward grade trend and increasing sub-point hits as months progress.

Progression views:

Cumulative 3-D voxel cube per month (snapshots + animated GIF).

Line charts: monthly average grade index per competency and overall.

Data: raw simulated entries saved to CSV.

How to tweak

Inside the script you can change:

n_entries = 50         # number of input records
months = 12            # tracking window
trend_strength = 0.35  # increase for faster improvement over time

grades = [f"G{i}" for i in range(1, 10)]
competencies = ["Tech Knowledge", "Mentoring", "Business", "Growth"]
subpoints = [f"S{i}" for i in range(1, 6)]


You can also plug in real user entries (date, competency, subpoint, grade) by loading a CSV instead of simulation and all plots/GIFs will regenerate.

Steps (so you can re-run elsewhere)

Generate/supply entries with (timestamp, competency, subpoint, grade_idx).

Aggregate by month → compute avg grade per competency and overall.

Build cumulative voxel grid per month → export snapshots & GIF.

Plot line charts per competency + overall.

Assumptions

Grades are treated as indices 0..8 (displayed as G1..G9).

Progression charts use monthly average grade (you could switch to max or median easily).

Risks / notes

Occlusion: Dense front-layer voxels can hide back layers. The GIF helps by showing cumulative growth over time.

Sampling bias: If some months have too few entries, averages can wobble. The program weights by actual hits.
