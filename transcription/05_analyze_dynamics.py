#!/usr/bin/env python3
"""Analyze board meeting power dynamics and treatment patterns.

Queries meeting_dynamics metadata stored by 04_ingest_transcripts.py,
computes cross-meeting statistical comparisons for a target board member
vs peers, detects trends, and generates visualizations.

Measurable treatment signals:
  1. Floor control: speaking time share, turn length, interruptions
  2. Response latency: how quickly others reply to a speaker
  3. Network centrality: who interacts with whom
  4. Acoustic profile: speaking pace, confidence patterns

Usage:
    python 05_analyze_dynamics.py "Donald Cook"
    python 05_analyze_dynamics.py "Donald Cook" --output report.html
    python 05_analyze_dynamics.py --list-speakers
    python 05_analyze_dynamics.py "Donald Cook" --date-from 2023-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from textwrap import dedent

import asyncpg
import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats as scipy_stats

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TENANT = "kent_sd"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output"

# Minimum meetings a speaker must appear in to be included in peer comparison
MIN_MEETINGS_FOR_PEER = 3


def _set_min_meetings(n: int) -> None:
    """Update the minimum meetings threshold (called from CLI)."""
    global MIN_MEETINGS_FOR_PEER  # noqa: PLW0603
    MIN_MEETINGS_FOR_PEER = n


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

FETCH_TRANSCRIPTS_SQL = """
SELECT id, title, meeting_date, committee_name, metadata
FROM documents
WHERE tenant_id = $1
  AND document_type = 'transcript'
  AND processing_status = 'complete'
  AND metadata ? 'meeting_dynamics'
ORDER BY meeting_date
"""


@dataclass
class MeetingRecord:
    """One meeting's dynamics data."""

    doc_id: str
    title: str
    meeting_date: date
    committee_name: str
    speaker_dynamics: dict[str, dict]
    interaction_pairs: list[dict]
    total_speaking_time_s: float
    total_turns: int
    speaker_map: dict[str, str]


async def load_meetings(
    dsn: str,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[MeetingRecord]:
    """Load all transcript documents with meeting_dynamics metadata."""
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(FETCH_TRANSCRIPTS_SQL, TENANT)
    finally:
        await conn.close()

    meetings = []
    for row in rows:
        md = row["meeting_date"]
        if date_from and md < date_from:
            continue
        if date_to and md > date_to:
            continue

        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)

        dynamics = meta.get("meeting_dynamics", {})
        if not dynamics:
            continue

        meetings.append(
            MeetingRecord(
                doc_id=str(row["id"]),
                title=row["title"],
                meeting_date=md,
                committee_name=row["committee_name"] or "",
                speaker_dynamics=dynamics.get("speaker_dynamics", {}),
                interaction_pairs=dynamics.get("interaction_pairs", []),
                total_speaking_time_s=dynamics.get("total_speaking_time_s", 0),
                total_turns=dynamics.get("total_turns", 0),
                speaker_map=meta.get("speaker_map", {}),
            )
        )

    return meetings


# ---------------------------------------------------------------------------
# Speaker name normalization
# ---------------------------------------------------------------------------

# Board members we want to track — canonical names
KNOWN_BOARD_MEMBERS = [
    "Meghin Margel",
    "Donald Cook",
    "Hyun-Jin Song",
    "Tim Clark",
    "Joe Farah",
    "Awale Farah",
    "Leslie Hamada",
    "Maya Vengadasalam",
    "Denise Gregory",
    "Nyema Williams",
    "Israel Vela",
]


def normalize_speaker_name(name: str) -> str | None:
    """Match a speaker label to a known board member name.

    Returns the canonical name if matched, None otherwise.
    Handles partial matches and common variations.
    """
    name_lower = name.lower().strip()

    for known in KNOWN_BOARD_MEMBERS:
        if known.lower() == name_lower:
            return known
        # Last-name match (e.g., "Cook" → "Donald Cook")
        parts = known.lower().split()
        if any(part == name_lower for part in parts):
            return known
        # Check if name contains the known name
        if known.lower() in name_lower or name_lower in known.lower():
            return known

    return None


def resolve_meeting_speakers(meeting: MeetingRecord) -> dict[str, str]:
    """Build a mapping from dynamics speaker labels to canonical names.

    Uses the speaker_map (from LLM resolution) and name normalization.
    """
    label_to_canonical: dict[str, str] = {}

    # First, use the speaker_map from LLM resolution
    reverse_map: dict[str, str] = {}
    for raw_label, resolved_name in meeting.speaker_map.items():
        canonical = normalize_speaker_name(resolved_name)
        if canonical:
            reverse_map[resolved_name] = canonical

    # Now map dynamics speaker labels
    for spk_label in meeting.speaker_dynamics:
        # Direct canonical match
        canonical = normalize_speaker_name(spk_label)
        if canonical:
            label_to_canonical[spk_label] = canonical
            continue

        # Check if this label appears in reverse_map
        if spk_label in reverse_map:
            label_to_canonical[spk_label] = reverse_map[spk_label]

    return label_to_canonical


# ---------------------------------------------------------------------------
# Cross-meeting aggregation
# ---------------------------------------------------------------------------


@dataclass
class SpeakerProfile:
    """Aggregated metrics for one speaker across meetings."""

    name: str
    meetings: list[date] = field(default_factory=list)
    speaking_share_pct: list[float] = field(default_factory=list)
    turn_count: list[int] = field(default_factory=list)
    total_words: list[int] = field(default_factory=list)
    mean_turn_length_words: list[float] = field(default_factory=list)
    mean_wpm: list[float] = field(default_factory=list)
    mean_confidence: list[float] = field(default_factory=list)
    interruptions_suffered: list[int] = field(default_factory=list)
    interruptions_initiated: list[int] = field(default_factory=list)
    mean_response_latency_s: list[float] = field(default_factory=list)
    # Normalized interruption rate (per turn)
    interruption_rate_suffered: list[float] = field(default_factory=list)
    interruption_rate_initiated: list[float] = field(default_factory=list)


def build_speaker_profiles(
    meetings: list[MeetingRecord],
) -> dict[str, SpeakerProfile]:
    """Aggregate per-speaker metrics across all meetings."""
    profiles: dict[str, SpeakerProfile] = {}

    for meeting in meetings:
        label_map = resolve_meeting_speakers(meeting)

        for spk_label, stats in meeting.speaker_dynamics.items():
            canonical = label_map.get(spk_label)
            if not canonical:
                continue  # Skip unresolved speakers

            if canonical not in profiles:
                profiles[canonical] = SpeakerProfile(name=canonical)

            p = profiles[canonical]
            p.meetings.append(meeting.meeting_date)
            p.speaking_share_pct.append(stats.get("speaking_share_pct", 0))
            p.turn_count.append(stats.get("turn_count", 0))
            p.total_words.append(stats.get("total_words", 0))
            p.mean_turn_length_words.append(stats.get("mean_turn_length_words", 0))
            p.mean_wpm.append(stats.get("mean_wpm", 0))
            p.mean_confidence.append(stats.get("mean_confidence", 0))
            p.interruptions_suffered.append(stats.get("interruptions_suffered", 0))
            p.interruptions_initiated.append(stats.get("interruptions_initiated", 0))

            latency = stats.get("mean_response_latency_s")
            if latency is not None:
                p.mean_response_latency_s.append(latency)

            # Normalized interruption rate
            turns = stats.get("turn_count", 1) or 1
            p.interruption_rate_suffered.append(stats.get("interruptions_suffered", 0) / turns)
            p.interruption_rate_initiated.append(stats.get("interruptions_initiated", 0) / turns)

    return profiles


# ---------------------------------------------------------------------------
# Interaction network
# ---------------------------------------------------------------------------


@dataclass
class InteractionNetwork:
    """Speaker-to-speaker interaction graph across meetings."""

    # edges[from_speaker][to_speaker] = total count
    edges: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    # Per-meeting edges for trend analysis
    meeting_edges: list[tuple[date, dict[str, dict[str, int]]]] = field(default_factory=list)


def build_interaction_network(
    meetings: list[MeetingRecord],
) -> InteractionNetwork:
    """Build a cross-meeting interaction network from adjacency pairs."""
    network = InteractionNetwork()

    for meeting in meetings:
        label_map = resolve_meeting_speakers(meeting)
        meeting_net: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for pair in meeting.interaction_pairs:
            from_canonical = label_map.get(pair["from"])
            to_canonical = label_map.get(pair["to"])
            if from_canonical and to_canonical:
                count = pair["count"]
                network.edges[from_canonical][to_canonical] += count
                meeting_net[from_canonical][to_canonical] += count

        if meeting_net:
            network.meeting_edges.append((meeting.meeting_date, dict(meeting_net)))

    return network


def compute_network_centrality(
    network: InteractionNetwork,
) -> dict[str, dict[str, float]]:
    """Compute in-degree, out-degree, and weighted centrality per speaker."""
    centrality: dict[str, dict[str, float]] = {}
    all_speakers = set(network.edges.keys())
    for targets in network.edges.values():
        all_speakers.update(targets.keys())

    total_edges = sum(count for targets in network.edges.values() for count in targets.values())

    for spk in all_speakers:
        out_degree = sum(network.edges.get(spk, {}).values())
        in_degree = sum(targets.get(spk, 0) for targets in network.edges.values())
        centrality[spk] = {
            "out_degree": out_degree,
            "in_degree": in_degree,
            "total_degree": in_degree + out_degree,
            "out_share": out_degree / total_edges if total_edges else 0,
            "in_share": in_degree / total_edges if total_edges else 0,
        }

    return centrality


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing target speaker vs peers on one metric."""

    metric_name: str
    target_name: str
    target_mean: float
    target_median: float
    target_n: int
    peer_mean: float
    peer_median: float
    peer_n: int
    # Welch's t-test
    t_statistic: float | None = None
    t_pvalue: float | None = None
    # Mann-Whitney U
    u_statistic: float | None = None
    u_pvalue: float | None = None
    # Effect size (Cohen's d)
    cohens_d: float | None = None
    # Direction: "higher", "lower", or "similar"
    direction: str = "similar"
    significant: bool = False


def compare_target_vs_peers(
    target: SpeakerProfile,
    peers: list[SpeakerProfile],
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Run statistical tests comparing target speaker against all peers.

    Tests each metric with Welch's t-test and Mann-Whitney U.
    Applies Holm-Bonferroni correction for multiple comparisons.
    """
    metrics = [
        ("Speaking Time Share (%)", "speaking_share_pct"),
        ("Mean Turn Length (words)", "mean_turn_length_words"),
        ("Speaking Pace (WPM)", "mean_wpm"),
        ("Confidence Score", "mean_confidence"),
        ("Interruptions Suffered (per turn)", "interruption_rate_suffered"),
        ("Interruptions Initiated (per turn)", "interruption_rate_initiated"),
        ("Response Latency (s)", "mean_response_latency_s"),
    ]

    results: list[ComparisonResult] = []

    for metric_name, attr in metrics:
        target_vals = np.array(getattr(target, attr), dtype=float)
        # Pool all peer values
        peer_vals = np.concatenate([np.array(getattr(p, attr), dtype=float) for p in peers])

        if len(target_vals) < 2 or len(peer_vals) < 2:
            results.append(
                ComparisonResult(
                    metric_name=metric_name,
                    target_name=target.name,
                    target_mean=float(np.mean(target_vals)) if len(target_vals) else 0,
                    target_median=float(np.median(target_vals)) if len(target_vals) else 0,
                    target_n=len(target_vals),
                    peer_mean=float(np.mean(peer_vals)) if len(peer_vals) else 0,
                    peer_median=float(np.median(peer_vals)) if len(peer_vals) else 0,
                    peer_n=len(peer_vals),
                )
            )
            continue

        # Welch's t-test (unequal variance)
        t_stat, t_pval = scipy_stats.ttest_ind(target_vals, peer_vals, equal_var=False)

        # Mann-Whitney U (non-parametric)
        try:
            u_stat, u_pval = scipy_stats.mannwhitneyu(target_vals, peer_vals, alternative="two-sided")
        except ValueError:
            u_stat, u_pval = None, None

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(target_vals, ddof=1) + np.var(peer_vals, ddof=1)) / 2)
        cohens_d = (np.mean(target_vals) - np.mean(peer_vals)) / pooled_std if pooled_std > 0 else 0

        direction = "similar"
        if np.mean(target_vals) > np.mean(peer_vals):
            direction = "higher"
        elif np.mean(target_vals) < np.mean(peer_vals):
            direction = "lower"

        results.append(
            ComparisonResult(
                metric_name=metric_name,
                target_name=target.name,
                target_mean=float(np.mean(target_vals)),
                target_median=float(np.median(target_vals)),
                target_n=len(target_vals),
                peer_mean=float(np.mean(peer_vals)),
                peer_median=float(np.median(peer_vals)),
                peer_n=len(peer_vals),
                t_statistic=float(t_stat),
                t_pvalue=float(t_pval),
                u_statistic=float(u_stat) if u_stat is not None else None,
                u_pvalue=float(u_pval) if u_pval is not None else None,
                cohens_d=float(cohens_d),
                direction=direction,
            )
        )

    # Holm-Bonferroni correction
    pvalues = [r.t_pvalue for r in results if r.t_pvalue is not None]
    if pvalues:
        sorted_indices = np.argsort(pvalues)
        m = len(pvalues)
        for rank, idx in enumerate(sorted_indices):
            adjusted_alpha = alpha / (m - rank)
            # Find the matching result
            pval_count = 0
            for r in results:
                if r.t_pvalue is not None:
                    if pval_count == idx:
                        r.significant = r.t_pvalue <= adjusted_alpha
                        break
                    pval_count += 1

    return results


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


@dataclass
class TrendResult:
    """Trend analysis for one metric over time."""

    metric_name: str
    speaker_name: str
    dates: list[date]
    values: list[float]
    rolling_mean: list[float]
    # Linear regression
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    slope_pvalue: float = 1.0
    trend_direction: str = "flat"


def compute_trends(
    profile: SpeakerProfile,
    window: int = 5,
) -> list[TrendResult]:
    """Compute time-series trends with rolling averages and linear regression."""
    metrics = [
        ("Speaking Time Share (%)", "speaking_share_pct"),
        ("Interruptions Suffered (per turn)", "interruption_rate_suffered"),
        ("Interruptions Initiated (per turn)", "interruption_rate_initiated"),
        ("Speaking Pace (WPM)", "mean_wpm"),
        ("Response Latency (s)", "mean_response_latency_s"),
    ]

    results = []
    for metric_name, attr in metrics:
        values = getattr(profile, attr)
        dates = profile.meetings[: len(values)]

        if len(values) < 3:
            continue

        vals = np.array(values, dtype=float)
        # Rolling mean
        if len(vals) >= window:
            rolling = pd.Series(vals).rolling(window, min_periods=1).mean().tolist()
        else:
            rolling = pd.Series(vals).expanding(min_periods=1).mean().tolist()

        # Linear regression (days since first meeting as x)
        x = np.array([(d - dates[0]).days for d in dates], dtype=float)
        slope, intercept, r_value, p_value, _ = scipy_stats.linregress(x, vals)

        direction = "flat"
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"

        results.append(
            TrendResult(
                metric_name=metric_name,
                speaker_name=profile.name,
                dates=dates,
                values=list(vals),
                rolling_mean=rolling,
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_value**2),
                slope_pvalue=float(p_value),
                trend_direction=direction,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_visualizations(
    target: SpeakerProfile,
    peers: list[SpeakerProfile],
    comparisons: list[ComparisonResult],
    trends: list[TrendResult],
    network: InteractionNetwork,
    centrality: dict[str, dict[str, float]],
    output_dir: Path,
) -> list[Path]:
    """Generate all visualization plots. Returns list of saved file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )

    # 1. Box plots: target vs peers for each metric
    saved.append(_plot_comparison_boxes(target, peers, comparisons, output_dir))

    # 2. Time series trends
    if trends:
        saved.append(_plot_trends(target, trends, output_dir))

    # 3. Interruption heatmap across meetings
    all_profiles = [target] + peers
    if len(all_profiles) >= 2:
        saved.append(_plot_interruption_matrix(all_profiles, output_dir))

    # 4. Network centrality bar chart
    if centrality:
        saved.append(_plot_centrality(centrality, target.name, output_dir))

    # 5. Speaking time share over time (all speakers)
    saved.append(_plot_speaking_time_trends(target, peers, output_dir))

    return [p for p in saved if p is not None]


def _plot_comparison_boxes(
    target: SpeakerProfile,
    peers: list[SpeakerProfile],
    comparisons: list[ComparisonResult],
    output_dir: Path,
) -> Path:
    """Box plots comparing target vs pooled peers for each metric."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Treatment Comparison: {target.name} vs Peers",
        fontsize=14,
        fontweight="bold",
    )
    axes = axes.flatten()

    metric_attrs = [
        ("Speaking Time\nShare (%)", "speaking_share_pct"),
        ("Mean Turn\nLength (words)", "mean_turn_length_words"),
        ("Speaking Pace\n(WPM)", "mean_wpm"),
        ("Confidence\nScore", "mean_confidence"),
        ("Interruptions\nSuffered/Turn", "interruption_rate_suffered"),
        ("Interruptions\nInitiated/Turn", "interruption_rate_initiated"),
        ("Response\nLatency (s)", "mean_response_latency_s"),
    ]

    for i, (label, attr) in enumerate(metric_attrs):
        ax = axes[i]
        target_vals = getattr(target, attr)
        peer_vals = []
        for p in peers:
            peer_vals.extend(getattr(p, attr))

        data = []
        labels_list = []
        if target_vals:
            data.append(target_vals)
            labels_list.append(target.name)
        if peer_vals:
            data.append(peer_vals)
            labels_list.append("Peers (pooled)")

        if data:
            bp = ax.boxplot(data, labels=labels_list, patch_artist=True)
            colors = ["#e74c3c", "#3498db"]
            for patch, color in zip(bp["boxes"], colors[: len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(label, fontsize=9)

        # Add significance marker
        comp = next(
            (c for c in comparisons if c.metric_name.startswith(label.split("\n")[0])),
            None,
        )
        if comp and comp.significant:
            ax.annotate(
                f"p={comp.t_pvalue:.4f} *",
                xy=(0.5, 0.95),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="red",
                fontweight="bold",
            )

    # Hide unused subplot
    axes[-1].set_visible(False)

    plt.tight_layout()
    path = output_dir / "comparison_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _plot_trends(
    target: SpeakerProfile,
    trends: list[TrendResult],
    output_dir: Path,
) -> Path:
    """Time series plots with rolling averages and linear fit."""
    n = len(trends)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Metric Trends Over Time: {target.name}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, trend in zip(axes, trends):
        dates_dt = [datetime.combine(d, datetime.min.time()) for d in trend.dates]

        ax.scatter(dates_dt, trend.values, alpha=0.4, s=20, color="#3498db", label="Per meeting")
        ax.plot(dates_dt, trend.rolling_mean, color="#e74c3c", linewidth=2, label="Rolling avg")

        # Linear trend line
        x_numeric = np.array([(d - trend.dates[0]).days for d in trend.dates])
        trend_line = trend.intercept + trend.slope * x_numeric
        ax.plot(dates_dt, trend_line, "--", color="#95a5a6", linewidth=1, label="Linear trend")

        ax.set_ylabel(trend.metric_name, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

        sig_str = ""
        if trend.slope_pvalue < 0.05:
            sig_str = f" (p={trend.slope_pvalue:.3f}, R²={trend.r_squared:.2f})"
        ax.set_title(
            f"{trend.metric_name}: {trend.trend_direction}{sig_str}",
            fontsize=10,
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.xticks(rotation=45)
    plt.tight_layout()
    path = output_dir / "trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _plot_interruption_matrix(
    profiles: list[SpeakerProfile],
    output_dir: Path,
) -> Path | None:
    """Heatmap of mean interruption rates across speakers."""
    names = [p.name for p in profiles if len(p.meetings) >= MIN_MEETINGS_FOR_PEER]
    if len(names) < 2:
        return None

    # Build matrix: rows = suffered, cols = speaker
    data = {}
    for p in profiles:
        if p.name not in names:
            continue
        data[p.name] = {
            "suffered": np.mean(p.interruption_rate_suffered) if p.interruption_rate_suffered else 0,
            "initiated": np.mean(p.interruption_rate_initiated) if p.interruption_rate_initiated else 0,
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(names) * 0.6)))
    fig.suptitle("Interruption Rates by Speaker", fontsize=14, fontweight="bold")

    # Suffered
    suffered = [data[n]["suffered"] for n in names]
    colors = ["#e74c3c" if s == max(suffered) else "#3498db" for s in suffered]
    ax1.barh(names, suffered, color=colors, alpha=0.7)
    ax1.set_xlabel("Mean Interruptions Suffered / Turn")
    ax1.set_title("Interrupted by Others")

    # Initiated
    initiated = [data[n]["initiated"] for n in names]
    colors = ["#e74c3c" if s == max(initiated) else "#3498db" for s in initiated]
    ax2.barh(names, initiated, color=colors, alpha=0.7)
    ax2.set_xlabel("Mean Interruptions Initiated / Turn")
    ax2.set_title("Interrupting Others")

    plt.tight_layout()
    path = output_dir / "interruption_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _plot_centrality(
    centrality: dict[str, dict[str, float]],
    target_name: str,
    output_dir: Path,
) -> Path:
    """Bar chart of network centrality (in-degree vs out-degree)."""
    # Sort by total degree
    speakers = sorted(centrality.keys(), key=lambda s: centrality[s]["total_degree"], reverse=True)[:15]  # Top 15

    fig, ax = plt.subplots(figsize=(14, max(5, len(speakers) * 0.5)))
    fig.suptitle("Conversation Network Centrality", fontsize=14, fontweight="bold")

    y_pos = np.arange(len(speakers))
    in_vals = [centrality[s]["in_degree"] for s in speakers]
    out_vals = [centrality[s]["out_degree"] for s in speakers]

    bar_height = 0.35
    ax.barh(
        y_pos - bar_height / 2,
        in_vals,
        bar_height,
        label="In-degree (spoken TO)",
        color="#3498db",
        alpha=0.7,
    )
    ax.barh(
        y_pos + bar_height / 2,
        out_vals,
        bar_height,
        label="Out-degree (spoke AFTER)",
        color="#e74c3c",
        alpha=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Total Interaction Count")
    ax.legend()

    # Highlight target speaker
    for i, spk in enumerate(speakers):
        if spk == target_name:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#e74c3c")

    plt.tight_layout()
    path = output_dir / "network_centrality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _plot_speaking_time_trends(
    target: SpeakerProfile,
    peers: list[SpeakerProfile],
    output_dir: Path,
) -> Path:
    """Speaking time share over time for target and each peer."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Speaking Time Share Over Time", fontsize=14, fontweight="bold")

    # Plot target prominently
    if target.meetings:
        dates_dt = [datetime.combine(d, datetime.min.time()) for d in target.meetings]
        ax.plot(
            dates_dt,
            target.speaking_share_pct,
            "o-",
            linewidth=2.5,
            markersize=6,
            label=target.name,
            color="#e74c3c",
            zorder=10,
        )

    # Plot peers with lighter lines
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(peers), 1)))
    for peer, color in zip(peers, colors):
        if len(peer.meetings) < MIN_MEETINGS_FOR_PEER:
            continue
        dates_dt = [datetime.combine(d, datetime.min.time()) for d in peer.meetings]
        ax.plot(
            dates_dt,
            peer.speaking_share_pct,
            ".-",
            linewidth=1,
            markersize=3,
            label=peer.name,
            color=color,
            alpha=0.6,
        )

    ax.set_ylabel("Speaking Time Share (%)")
    ax.set_xlabel("Meeting Date")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = output_dir / "speaking_time_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_text_report(
    target: SpeakerProfile,
    peers: list[SpeakerProfile],
    comparisons: list[ComparisonResult],
    trends: list[TrendResult],
    centrality: dict[str, dict[str, float]],
    meetings: list[MeetingRecord],
) -> str:
    """Generate a plain-text statistical report."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"BOARD MEETING DYNAMICS ANALYSIS: {target.name}")
    lines.append(f"Meetings analyzed: {len(meetings)}")
    lines.append(f"Date range: {meetings[0].meeting_date} to {meetings[-1].meeting_date}")
    lines.append(f"Target appearances: {len(target.meetings)}")
    peer_names = [p.name for p in peers if len(p.meetings) >= MIN_MEETINGS_FOR_PEER]
    lines.append(f"Peers (≥{MIN_MEETINGS_FOR_PEER} meetings): {', '.join(peer_names)}")
    lines.append("=" * 72)

    # Section 1: Statistical comparisons
    lines.append("")
    lines.append("1. STATISTICAL COMPARISON: TARGET vs PEERS")
    lines.append("-" * 50)
    lines.append(f"{'Metric':<35} {'Target':>8} {'Peers':>8} {'p-value':>8} {'Effect':>8} {'Sig':>4}")
    lines.append("-" * 80)

    for comp in comparisons:
        sig_marker = " **" if comp.significant else ""
        pval_str = f"{comp.t_pvalue:.4f}" if comp.t_pvalue is not None else "N/A"
        d_str = f"{comp.cohens_d:+.2f}" if comp.cohens_d is not None else "N/A"
        lines.append(
            f"{comp.metric_name:<35} {comp.target_mean:>8.2f} "
            f"{comp.peer_mean:>8.2f} {pval_str:>8} {d_str:>8}{sig_marker}"
        )

    lines.append("")
    lines.append("  ** = significant after Holm-Bonferroni correction")
    lines.append("  Effect = Cohen's d (positive = target higher than peers)")

    # Section 2: Trend analysis
    if trends:
        lines.append("")
        lines.append(f"2. TREND ANALYSIS: {target.name}")
        lines.append("-" * 50)
        for trend in trends:
            sig = " *" if trend.slope_pvalue < 0.05 else ""
            lines.append(
                f"  {trend.metric_name:<40} {trend.trend_direction:>12} "
                f"(slope={trend.slope:+.4f}, p={trend.slope_pvalue:.3f}, "
                f"R²={trend.r_squared:.3f}){sig}"
            )

    # Section 3: Network centrality
    if centrality:
        lines.append("")
        lines.append("3. NETWORK CENTRALITY")
        lines.append("-" * 50)
        lines.append(f"{'Speaker':<25} {'In-deg':>8} {'Out-deg':>8} {'Total':>8}")
        lines.append("-" * 55)

        sorted_speakers = sorted(
            centrality.keys(),
            key=lambda s: centrality[s]["total_degree"],
            reverse=True,
        )
        for spk in sorted_speakers[:10]:
            c = centrality[spk]
            marker = " <<<" if spk == target.name else ""
            lines.append(
                f"{spk:<25} {c['in_degree']:>8.0f} {c['out_degree']:>8.0f} " f"{c['total_degree']:>8.0f}{marker}"
            )

    # Section 4: Per-meeting breakdown for target
    lines.append("")
    lines.append(f"4. PER-MEETING BREAKDOWN: {target.name}")
    lines.append("-" * 50)
    lines.append(f"{'Date':<12} {'Share%':>7} {'Turns':>6} {'Words':>6} " f"{'WPM':>5} {'Int.Suf':>8} {'Int.Init':>8}")
    lines.append("-" * 65)

    for i, d in enumerate(target.meetings):
        lines.append(
            f"{d.isoformat():<12} "
            f"{target.speaking_share_pct[i]:>7.1f} "
            f"{target.turn_count[i]:>6} "
            f"{target.total_words[i]:>6} "
            f"{target.mean_wpm[i]:>5.0f} "
            f"{target.interruptions_suffered[i]:>8} "
            f"{target.interruptions_initiated[i]:>8}"
        )

    # Section 5: Interpretation notes
    lines.append("")
    lines.append("5. INTERPRETATION NOTES")
    lines.append("-" * 50)
    lines.append("  - Correlation ≠ discrimination: higher interruption rates may")
    lines.append("    stem from procedural roles (e.g., chair moderating) rather")
    lines.append("    than bias.")
    lines.append("  - Context matters: negative tone during budget crises may not")
    lines.append("    reflect personal animus.")
    lines.append("  - Interruption proxy: uses <0.3s speaker-change gap (WhisperX")
    lines.append("    segments do not truly overlap).")
    lines.append("  - Speaker resolution quality depends on LLM accuracy; verify")
    lines.append("    speaker_map in document metadata for key meetings.")
    lines.append("")

    return "\n".join(lines)


def generate_html_report(
    text_report: str,
    image_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Generate an HTML report with embedded images."""
    import base64

    images_html = []
    for img_path in image_paths:
        if img_path.exists():
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            images_html.append(
                f'<div class="chart"><img src="data:image/png;base64,{b64}" ' f'alt="{img_path.stem}"></div>'
            )

    html = dedent(f"""\
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Board Meeting Dynamics Analysis</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                   max-width: 1200px; margin: 0 auto; padding: 20px;
                   background: #f5f5f5; }}
            pre {{ background: #1a1a2e; color: #e0e0e0; padding: 20px;
                  border-radius: 8px; overflow-x: auto; font-size: 13px;
                  line-height: 1.5; }}
            .chart {{ margin: 20px 0; text-align: center; }}
            .chart img {{ max-width: 100%; border-radius: 8px;
                         box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db;
                 padding-bottom: 10px; }}
            .disclaimer {{ background: #ffeaa7; padding: 15px;
                          border-radius: 8px; margin: 20px 0;
                          border-left: 4px solid #fdcb6e; }}
        </style>
    </head>
    <body>
        <h1>Board Meeting Dynamics Analysis</h1>
        <div class="disclaimer">
            <strong>Interpretation note:</strong> Statistical patterns do not
            establish causation or intent.  Higher interruption rates may
            reflect procedural roles, topic complexity, or meeting dynamics
            rather than bias.  Always review specific meeting context.
        </div>
        <h2>Statistical Report</h2>
        <pre>{text_report}</pre>
        <h2>Visualizations</h2>
        {''.join(images_html)}
    </body>
    </html>
    """)

    path = output_dir / "dynamics_report.html"
    path.write_text(html)
    logger.info("Saved HTML report: %s", path)
    return path


# ---------------------------------------------------------------------------
# Speaker listing
# ---------------------------------------------------------------------------


async def list_speakers(dsn: str) -> None:
    """Print all resolved speaker names found across transcript meetings."""
    meetings = await load_meetings(dsn)
    if not meetings:
        print("No transcript meetings found in database.")
        return

    speaker_counts: dict[str, int] = defaultdict(int)
    for meeting in meetings:
        label_map = resolve_meeting_speakers(meeting)
        for canonical in set(label_map.values()):
            speaker_counts[canonical] += 1

    print(f"\nResolved speakers across {len(meetings)} meetings:")
    print(f"{'Speaker':<30} {'Meetings':>8}")
    print("-" * 40)
    for name, count in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<30} {count:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_dsn() -> str:
    """Build PostgreSQL DSN from environment variables."""
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "qorvault")
    user = os.environ.get("POSTGRES_USER", "qorvault")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD not set. Copy .env.example to .env and fill in.")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def run_analysis(
    target_name: str,
    date_from: date | None = None,
    date_to: date | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Run the full analysis pipeline."""
    dsn = get_dsn()

    # Load meetings
    logger.info("Loading transcript meetings...")
    meetings = await load_meetings(dsn, date_from, date_to)
    if not meetings:
        logger.error("No transcript meetings found. Run 04_ingest_transcripts.py first.")
        return

    logger.info("Loaded %d meetings (%s to %s)", len(meetings), meetings[0].meeting_date, meetings[-1].meeting_date)

    # Build speaker profiles
    profiles = build_speaker_profiles(meetings)
    logger.info("Found %d resolved speakers", len(profiles))

    # Find target
    target = profiles.get(target_name)
    if not target:
        # Try fuzzy match
        for name, profile in profiles.items():
            if target_name.lower() in name.lower():
                target = profile
                logger.info("Matched target: '%s' → '%s'", target_name, name)
                break

    if not target:
        logger.error(
            "Target speaker '%s' not found. Available speakers: %s",
            target_name,
            ", ".join(sorted(profiles.keys())),
        )
        return

    # Build peer list (everyone except target with enough meetings)
    peers = [p for name, p in profiles.items() if name != target.name and len(p.meetings) >= MIN_MEETINGS_FOR_PEER]
    if not peers:
        # Fall back to including all other speakers
        peers = [p for name, p in profiles.items() if name != target.name]
        logger.warning(
            "No peers with ≥%d meetings; using all %d other speakers",
            MIN_MEETINGS_FOR_PEER,
            len(peers),
        )

    logger.info(
        "Analyzing: %s (%d meetings) vs %d peers",
        target.name,
        len(target.meetings),
        len(peers),
    )

    # Statistical comparisons
    comparisons = compare_target_vs_peers(target, peers)

    # Trend analysis
    trends = compute_trends(target)

    # Network analysis
    network = build_interaction_network(meetings)
    centrality = compute_network_centrality(network)

    # Generate text report
    text_report = generate_text_report(target, peers, comparisons, trends, centrality, meetings)
    print(text_report)

    # Save text report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    report_path.write_text(text_report)
    logger.info("Saved text report: %s", report_path)

    # Generate visualizations
    image_paths = generate_visualizations(target, peers, comparisons, trends, network, centrality, output_dir)

    # Generate HTML report
    generate_html_report(text_report, image_paths, output_dir)

    # Export raw data as JSON for further analysis
    export = {
        "target": target.name,
        "meetings_analyzed": len(meetings),
        "date_range": [
            meetings[0].meeting_date.isoformat(),
            meetings[-1].meeting_date.isoformat(),
        ],
        "comparisons": [
            {
                "metric": c.metric_name,
                "target_mean": c.target_mean,
                "target_median": c.target_median,
                "target_n": c.target_n,
                "peer_mean": c.peer_mean,
                "peer_median": c.peer_median,
                "peer_n": c.peer_n,
                "t_pvalue": c.t_pvalue,
                "u_pvalue": c.u_pvalue,
                "cohens_d": c.cohens_d,
                "direction": c.direction,
                "significant": c.significant,
            }
            for c in comparisons
        ],
        "trends": [
            {
                "metric": t.metric_name,
                "direction": t.trend_direction,
                "slope": t.slope,
                "p_value": t.slope_pvalue,
                "r_squared": t.r_squared,
            }
            for t in trends
        ],
        "centrality": {spk: vals for spk, vals in centrality.items()},
        "per_meeting": [
            {
                "date": d.isoformat(),
                "speaking_share_pct": target.speaking_share_pct[i],
                "turn_count": target.turn_count[i],
                "total_words": target.total_words[i],
                "mean_wpm": target.mean_wpm[i],
                "interruptions_suffered": target.interruptions_suffered[i],
                "interruptions_initiated": target.interruptions_initiated[i],
            }
            for i, d in enumerate(target.meetings)
        ],
    }
    json_path = output_dir / "analysis_data.json"
    json_path.write_text(json.dumps(export, indent=2))
    logger.info("Saved JSON export: %s", json_path)

    print(f"\nOutputs saved to: {output_dir}/")
    print("  - report.txt (text report)")
    print("  - dynamics_report.html (HTML with charts)")
    print("  - analysis_data.json (raw data for further analysis)")
    for p in image_paths:
        print(f"  - {p.name}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze board meeting power dynamics and treatment patterns",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Target speaker name to analyze (e.g., 'Donald Cook')",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List all resolved speakers and exit",
    )
    parser.add_argument(
        "--date-from",
        type=lambda s: date.fromisoformat(s),
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        type=lambda s: date.fromisoformat(s),
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-meetings",
        type=int,
        default=MIN_MEETINGS_FOR_PEER,
        help=f"Minimum meetings for peer inclusion (default: {MIN_MEETINGS_FOR_PEER})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stderr,
    )

    _set_min_meetings(args.min_meetings)

    if args.list_speakers:
        dsn = get_dsn()
        asyncio.run(list_speakers(dsn))
        return 0

    if not args.target:
        parser.error("target speaker name required (or use --list-speakers)")

    asyncio.run(
        run_analysis(
            target_name=args.target,
            date_from=args.date_from,
            date_to=args.date_to,
            output_dir=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
