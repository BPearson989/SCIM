"""SCIM: Simulate competency improvement metrics and visualize them in 3-D.

This script creates a voxel visualization of student progression across
grades, competencies, and sub-points. It can either simulate input data or
load a user-provided CSV. The outputs include:

* Monthly 3-D voxel snapshots that highlight every (grade, competency,
  sub-point) combination that has been satisfied at least once up to that
  month.
* An animated GIF that stitches the monthly voxel snapshots together to
  illustrate progression over time.
* Line charts that show the monthly average grade index per competency and
  overall.
* A CSV export of the raw events used to generate the analytics.

The voxel visualization follows the convention described in the repository
README:

* X-axis → Grades G1–G9 (rendered back to front in the final view).
* Y-axis → Competencies (Tech Knowledge, Mentoring, Business, Growth).
* Z-axis → Sub-points S1–S5.

Example usage (with deterministic output):

```
python scim.py --seed 7
```

To load a CSV instead of simulation:

```
python scim.py --input my_entries.csv
```

Expected CSV columns: ``timestamp`` (ISO 8601 date), ``competency``,
``subpoint`` and either ``grade`` (e.g. ``G4``) or ``grade_idx`` (0-based
integer). The script will derive any missing representation automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import argparse
import importlib.util

import matplotlib


# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGEIO_SPEC = importlib.util.find_spec("imageio")
if IMAGEIO_SPEC is not None:
    import imageio.v2 as imageio  # type: ignore[assignment]
else:
    imageio = None


GRADES: List[str] = [f"G{i}" for i in range(1, 10)]
COMPETENCIES: List[str] = ["Tech Knowledge", "Mentoring", "Business", "Growth"]
SUBPOINTS: List[str] = [f"S{i}" for i in range(1, 6)]


@dataclass
class SimulationConfig:
    """Parameters that drive the synthetic data generation."""

    n_entries: int = 50
    months: int = 12
    trend_strength: float = 0.35
    seed: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate or visualize SCIM data")
    parser.add_argument(
        "--entries",
        type=int,
        default=50,
        help="Number of simulated records (ignored when --input is provided)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months to span in the simulation (ignored when loading CSV)",
    )
    parser.add_argument(
        "--trend-strength",
        type=float,
        default=0.35,
        help="How quickly grades trend upward over time (0..1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV to load instead of simulating data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory where charts, GIFs, and CSV exports will be written",
    )
    return parser.parse_args()


def simulate_entries(config: SimulationConfig) -> pd.DataFrame:
    """Generate synthetic entries that trend upward across months."""

    rng = np.random.default_rng(config.seed)
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(months=config.months - 1)

    records = []
    # We skew the distribution so later months receive more events, reflecting growth.
    for _ in range(config.n_entries):
        # Triangular distribution biased toward the most recent month (mode at months-1).
        month_offset = int(
            np.round(rng.triangular(left=0, mode=config.months - 1, right=config.months - 1))
        )
        month_offset = int(np.clip(month_offset, 0, config.months - 1))
        month_start = start + pd.DateOffset(months=month_offset)
        day = rng.integers(0, 28)
        timestamp = month_start + pd.Timedelta(days=int(day))

        # Trend grades upward through the year with controlled noise.
        trend_position = month_offset / max(config.months - 1, 1)
        mean_grade = trend_position * 8 * (1 + config.trend_strength)
        grade_idx = int(np.clip(rng.normal(loc=mean_grade, scale=1.5), 0, 8))

        competency_idx = rng.integers(0, len(COMPETENCIES))
        subpoint_idx = rng.integers(0, len(SUBPOINTS))

        records.append(
            {
                "timestamp": timestamp,
                "competency": COMPETENCIES[competency_idx],
                "subpoint": SUBPOINTS[subpoint_idx],
                "grade_idx": grade_idx,
            }
        )

    df = pd.DataFrame(records)
    df["grade"] = df["grade_idx"].map(lambda idx: GRADES[idx])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_entries_from_csv(path: Path) -> pd.DataFrame:
    """Load entries from a CSV file, validating and normalizing the schema."""

    df = pd.read_csv(path, parse_dates=["timestamp"])  # type: ignore[arg-type]
    missing_columns = {"timestamp", "competency", "subpoint"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    if "grade_idx" not in df.columns and "grade" not in df.columns:
        raise ValueError("CSV must include either 'grade_idx' or 'grade' column")

    if "grade_idx" not in df.columns:
        df["grade_idx"] = df["grade"].map(normalize_grade_label)
    else:
        df["grade_idx"] = df["grade_idx"].astype(int)

    if "grade" not in df.columns:
        df["grade"] = df["grade_idx"].map(lambda idx: GRADES[int(idx)])

    return df.sort_values("timestamp").reset_index(drop=True)


def normalize_grade_label(label: str) -> int:
    if isinstance(label, str) and label.startswith("G"):
        value = int(label[1:]) - 1
        if 0 <= value < len(GRADES):
            return value
    raise ValueError(f"Invalid grade label: {label}")


def prepare_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["timestamp"].dt.to_period("M").dt.to_timestamp()
    out["competency_idx"] = out["competency"].map(COMPETENCIES.index)
    out["subpoint_idx"] = out["subpoint"].map(SUBPOINTS.index)
    return out


def build_monthly_voxels(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, np.ndarray]]:
    """Create a voxel grid for each month showing cumulative progression."""

    months = np.sort(df["month"].unique())
    snapshots: List[Tuple[pd.Timestamp, np.ndarray]] = []

    for month in months:
        mask = df["month"] <= month
        active = df.loc[mask, ["grade_idx", "competency_idx", "subpoint_idx"]]
        voxels = np.zeros((len(GRADES), len(COMPETENCIES), len(SUBPOINTS)), dtype=bool)
        voxels[active["grade_idx"], active["competency_idx"], active["subpoint_idx"]] = True
        snapshots.append((pd.Timestamp(month), voxels))

    return snapshots


def render_voxel_snapshot(voxels: np.ndarray, title: str, output_path: Path) -> None:
    """Render a single voxel snapshot using Matplotlib."""

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Matplotlib's voxel renderer uses the x-axis as the first dimension. We flip
    # the grades so G1 appears at the back of the plot.
    flipped_voxels = np.flip(voxels, axis=0)

    # Allocate colors (RGBA) where True voxels receive a golden color.
    colors = np.zeros(flipped_voxels.shape + (4,), dtype=float)
    colors[flipped_voxels] = (0.91, 0.6, 0.1, 0.9)

    ax.voxels(
        flipped_voxels,
        facecolors=colors,
        edgecolor=(0.2, 0.2, 0.2, 0.4),
        linewidth=0.5,
    )

    ax.set_title(title)
    ax.set_xlabel("Grades (back → front)")
    ax.set_ylabel("Competencies (left → right)")
    ax.set_zlabel("Sub-points")

    # Tick labels align to the center of each voxel. We reverse the grade labels
    # to match the flipped array.
    ax.set_xticks(np.arange(len(GRADES)) + 0.5)
    ax.set_xticklabels(list(reversed(GRADES)))
    ax.set_yticks(np.arange(len(COMPETENCIES)) + 0.5)
    ax.set_yticklabels(COMPETENCIES)
    ax.set_zticks(np.arange(len(SUBPOINTS)) + 0.5)
    ax.set_zticklabels(SUBPOINTS)

    ax.set_box_aspect((len(GRADES), len(COMPETENCIES), len(SUBPOINTS)))
    ax.view_init(elev=25, azim=-70)

    # Add subtle grid lines to echo the reference visualization.
    _add_reference_grid(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _add_reference_grid(ax: matplotlib.axes.Axes) -> None:
    # Draw thin lines to emphasize the 3-D lattice.
    for x in range(len(GRADES) + 1):
        ax.plot([x, x], [0, len(COMPETENCIES)], [0, 0], color="#d0d0d0", linewidth=0.5, alpha=0.5)
        ax.plot([x, x], [0, 0], [0, len(SUBPOINTS)], color="#d0d0d0", linewidth=0.5, alpha=0.5)
    for y in range(len(COMPETENCIES) + 1):
        ax.plot([0, len(GRADES)], [y, y], [0, 0], color="#d0d0d0", linewidth=0.5, alpha=0.5)
        ax.plot([0, 0], [y, y], [0, len(SUBPOINTS)], color="#d0d0d0", linewidth=0.5, alpha=0.5)
    for z in range(len(SUBPOINTS) + 1):
        ax.plot([0, 0], [0, len(COMPETENCIES)], [z, z], color="#d0d0d0", linewidth=0.5, alpha=0.5)
        ax.plot([0, len(GRADES)], [0, 0], [z, z], color="#d0d0d0", linewidth=0.5, alpha=0.5)


def render_voxel_progression(
    snapshots: Sequence[Tuple[pd.Timestamp, np.ndarray]], output_dir: Path
) -> List[Tuple[pd.Timestamp, Path]]:
    rendered: List[Tuple[pd.Timestamp, Path]] = []
    for timestamp, voxels in snapshots:
        title = timestamp.strftime("Early profile (%b %Y)")
        path = output_dir / f"voxels_{timestamp.strftime('%Y_%m')}.png"
        render_voxel_snapshot(voxels, title, path)
        rendered.append((timestamp, path))
    return rendered


def create_voxel_timeline(
    frames: Sequence[Tuple[pd.Timestamp, Path]],
    output_path: Path,
    columns: int = 4,
) -> None:
    """Combine individual voxel snapshots into a single timeline image."""

    if not frames:
        return

    columns = max(1, int(columns))
    rows = int(np.ceil(len(frames) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.6, rows * 3.6))
    axes_array = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_array):
        if idx >= len(frames):
            ax.axis("off")
            continue

        timestamp, frame_path = frames[idx]
        image = plt.imread(frame_path)
        ax.imshow(image)
        ax.set_title(timestamp.strftime("%b %Y"))
        ax.axis("off")

    fig.subplots_adjust(wspace=0.05, hspace=0.2, top=0.88)
    fig.suptitle("Voxel progression timeline", fontsize=16, y=0.98)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_progress_gif(frames: Sequence[Path], output_path: Path, duration: float = 0.8) -> None:
    if imageio is None:
        print(
            "Skipping GIF creation because the optional 'imageio' dependency is not installed."
        )
        return

    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output_path, images, duration=duration)


def plot_competency_trends(df: pd.DataFrame, output_path: Path) -> None:
    monthly = (
        df.groupby(["month", "competency"], observed=True)["grade_idx"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    overall = df.groupby("month")["grade_idx"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    index = monthly.index
    if isinstance(index, pd.PeriodIndex):
        months = index.to_timestamp()
    else:
        months = pd.to_datetime(index)
    for competency in COMPETENCIES:
        ax.plot(months, monthly[competency], marker="o", label=competency)
    ax.plot(months, overall, marker="o", linestyle="--", color="black", label="Overall avg")

    ax.set_title("Monthly average grade index per competency")
    ax.set_ylabel("Average grade index (0=G1, 8=G9)")
    ax.set_xlabel("Month")
    ax.set_ylim(0, 8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_entries(df: pd.DataFrame, output_path: Path) -> None:
    columns = ["timestamp", "competency", "subpoint", "grade", "grade_idx"]
    df.to_csv(output_path, columns=columns, index=False)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        entries = load_entries_from_csv(args.input)
    else:
        config = SimulationConfig(
            n_entries=args.entries,
            months=args.months,
            trend_strength=args.trend_strength,
            seed=args.seed,
        )
        entries = simulate_entries(config)

    prepared = prepare_month_columns(entries)
    snapshots = build_monthly_voxels(prepared)
    frame_info = render_voxel_progression(snapshots, output_dir)
    if frame_info:
        create_progress_gif([path for _, path in frame_info], output_dir / "voxel_progress.gif")
        create_voxel_timeline(frame_info, output_dir / "voxel_timeline.png")

    plot_competency_trends(prepared, output_dir / "competency_trends.png")
    export_entries(prepared, output_dir / "entries.csv")

    print(f"Generated {len(frame_info)} voxel snapshots in {output_dir}")
    print(f"Saved animated GIF to {output_dir / 'voxel_progress.gif'}")
    if frame_info:
        print(f"Saved voxel timeline to {output_dir / 'voxel_timeline.png'}")
    print(f"Saved competency trend chart to {output_dir / 'competency_trends.png'}")
    print(f"Saved raw entries to {output_dir / 'entries.csv'}")


if __name__ == "__main__":
    main()

