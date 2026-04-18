#!/usr/bin/env python3
"""analyze_results.py — Parse NVBench JSON + latency CSV and generate a Markdown report with plots.

Usage:
  python analyze_results.py [--results-dir results/] [--output report.md]
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Plots will be skipped.", file=sys.stderr)


def parse_nvbench_json(filepath):
    """Parse NVBench JSON output and return a list of (type, elements, bandwidth_gb_s) tuples."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for bench in data.get("benchmarks", []):
        for state in bench.get("states", []):
            axes = {}
            for axis in state.get("axis_values", []):
                axes[axis["name"]] = axis["value"]

            type_name = axes.get("T{ct}", "unknown")
            elements = int(axes.get("Elements{io}", 0))

            bw_bytes_per_sec = 0.0
            for s in state.get("summaries", []):
                tag = s.get("tag", "")
                if tag == "nv/cold/bw/global/bytes_per_second":
                    for sd in s.get("data", []):
                        if sd["name"] == "value":
                            bw_bytes_per_sec = float(sd["value"])

            results.append(
                {
                    "type": type_name,
                    "elements": elements,
                    "bandwidth_gb_s": bw_bytes_per_sec / 1e9,
                }
            )
    return results


def parse_latency_csv(filepath):
    """Parse the latency harness CSV output."""
    results = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["benchmark"], row["metric"])
            results[key] = float(row["value_us"])
    return results


def plot_bandwidth(cub_results, aot_results, algorithm, output_dir):
    """Generate bandwidth comparison plot for a given algorithm."""
    if not HAS_MPL:
        return None

    types = sorted(set(r["type"] for r in cub_results))
    fig, axes = plt.subplots(1, len(types), figsize=(5 * len(types), 4), sharey=True)
    if len(types) == 1:
        axes = [axes]

    for ax, dtype in zip(axes, types):
        cub_pts = sorted(
            [
                (r["elements"], r["bandwidth_gb_s"])
                for r in cub_results
                if r["type"] == dtype
            ],
            key=lambda x: x[0],
        )
        aot_pts = sorted(
            [
                (r["elements"], r["bandwidth_gb_s"])
                for r in aot_results
                if r["type"] == dtype
            ],
            key=lambda x: x[0],
        )

        if cub_pts:
            ax.plot(
                [p[0] for p in cub_pts],
                [p[1] for p in cub_pts],
                "o-",
                label="CUB",
                markersize=3,
            )
        if aot_pts:
            ax.plot(
                [p[0] for p in aot_pts],
                [p[1] for p in aot_pts],
                "s--",
                label="AOT",
                markersize=3,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Elements")
        ax.set_title(f"{algorithm} — {dtype}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: (
                    f"2^{int(x).bit_length() - 1}"
                    if x > 0 and x == int(x) and (int(x) & (int(x) - 1)) == 0
                    else ""
                )
            )
        )

    axes[0].set_ylabel("Bandwidth (GB/s)")
    fig.suptitle(f"{algorithm}: CUB vs AOT Bandwidth", fontsize=14)
    fig.tight_layout()

    filename = f"bandwidth_{algorithm.lower()}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filename


def plot_latency_bars(latency_data, output_dir):
    """Generate latency comparison bar chart."""
    if not HAS_MPL:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, algo in [(ax1, "reduce"), (ax2, "transform")]:
        bars = {
            "Cold link": latency_data.get((f"aot_{algo}", "cold_link_us"), 0),
            "Hot link": latency_data.get((f"aot_{algo}", "hot_link_us"), 0),
        }
        labels = list(bars.keys())
        values = list(bars.values())
        colors = ["#d62728", "#2ca02c"]

        b = ax.barh(labels, values, color=colors)
        for bar, val in zip(b, values):
            ax.text(
                bar.get_width() + ax.get_xlim()[1] * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} us",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Latency (us)")
        ax.set_title(f"{algo.title()} Link Latency")
        ax.set_xlim(right=max(values) * 1.25)

    fig.suptitle("AOT Link Latency (int32, N=1M)", fontsize=14)
    fig.tight_layout()

    filename = "latency_comparison.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filename


def parse_first_vs_second_csv(filepath):
    """Parse first_vs_second.csv into {(kernel, position): [link_us, ...]}."""
    data = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["kernel"], row["position"])
            data.setdefault(key, []).append(float(row["link_us"]))
    return data


def plot_first_vs_second(data, output_dir):
    """Generate grouped bar chart: first-in-process vs second-in-process link latency."""
    if not HAS_MPL:
        return None

    # Determine kernel order: transforms then reduces, each sorted by type
    all_kernels = sorted(set(k for k, _ in data.keys()))
    transform_kernels = [k for k in all_kernels if k.startswith("transform_")]
    reduce_kernels = [k for k in all_kernels if k.startswith("reduce_")]
    kernels = transform_kernels + reduce_kernels

    # Compute stats, dropping run 1 (index 0) as OS cold-start outlier
    def trimmed(values):
        return sorted(values)[1:] if len(values) > 2 else values

    first_medians = []
    first_mins = []
    first_maxs = []
    second_medians = []
    second_mins = []
    second_maxs = []
    labels = []

    for k in kernels:
        algo, dtype = k.split("_", 1)
        labels.append(f"{algo}\n{dtype}")

        fvals = trimmed(data.get((k, "first"), [0]))
        svals = trimmed(data.get((k, "second"), [0]))

        first_medians.append(statistics.median(fvals))
        first_mins.append(min(fvals))
        first_maxs.append(max(fvals))
        second_medians.append(statistics.median(svals))
        second_mins.append(min(svals))
        second_maxs.append(max(svals))

    import numpy as np

    y = np.arange(len(kernels))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.barh(
        y + bar_height / 2,
        first_medians,
        bar_height,
        label="First in process",
        color="#d62728",
        alpha=0.85,
    )
    bars2 = ax.barh(
        y - bar_height / 2,
        second_medians,
        bar_height,
        label="Second in process",
        color="#2ca02c",
        alpha=0.85,
    )

    xmax = max(max(first_medians), max(second_medians)) * 1.65
    for bar, med, lo, hi in zip(bars1, first_medians, first_mins, first_maxs):
        ax.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"median: {med:.0f} (min: {lo:.0f}, max: {hi:.0f})",
            va="center",
            fontsize=8,
        )
    for bar, med, lo, hi in zip(bars2, second_medians, second_mins, second_maxs):
        ax.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"median: {med:.0f} (min: {lo:.0f}, max: {hi:.0f})",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Link latency (us)")
    ax.set_xlim(right=xmax)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(
        "nvJitLink Cold-Link Latency: First vs Second Call in Process\n"
        "(error bars = range across trials, excluding first trial)",
        fontsize=11,
    )
    fig.tight_layout()

    filename = "first_vs_second.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filename


def generate_report(results_dir, output_file):
    results_dir = Path(results_dir)
    output_dir = results_dir

    lines = ["# AOT Benchmark Report\n"]

    # Bandwidth results
    for algo in ["reduce", "transform"]:
        cub_file = results_dir / f"bench_{algo}_cub.json"
        aot_file = results_dir / f"bench_{algo}_aot.json"

        if not cub_file.exists() or not aot_file.exists():
            lines.append(f"\n## {algo.title()} Bandwidth\n")
            lines.append(f"*Data not found ({cub_file} or {aot_file})*\n")
            continue

        cub_results = parse_nvbench_json(cub_file)
        aot_results = parse_nvbench_json(aot_file)

        plot_file = plot_bandwidth(
            cub_results, aot_results, algo.title(), str(output_dir)
        )

        lines.append(f"\n## {algo.title()} Bandwidth\n")
        if plot_file:
            lines.append(f"![{algo} bandwidth]({plot_file})\n")

        # Summary table: peak bandwidth per type
        types = sorted(set(r["type"] for r in cub_results))
        lines.append(f"\n| Type | CUB Peak GB/s | AOT Peak GB/s | Ratio (AOT/CUB) |")
        lines.append(f"|------|--------------|--------------|-----------------|")
        for t in types:
            cub_peak = max(
                (r["bandwidth_gb_s"] for r in cub_results if r["type"] == t), default=0
            )
            aot_peak = max(
                (r["bandwidth_gb_s"] for r in aot_results if r["type"] == t), default=0
            )
            ratio = aot_peak / cub_peak if cub_peak > 0 else 0
            lines.append(f"| {t} | {cub_peak:.1f} | {aot_peak:.1f} | {ratio:.3f} |")
        lines.append("")

    # Latency results
    latency_file = results_dir / "latency.csv"
    if latency_file.exists():
        latency_data = parse_latency_csv(latency_file)
        plot_file = plot_latency_bars(latency_data, str(output_dir))

        lines.append("\n## Latency (int32, N=1M)\n")
        if plot_file:
            lines.append(f"![latency comparison]({plot_file})\n")

        lines.append("| Benchmark | Metric | Value (us) |")
        lines.append("|-----------|--------|-----------|")
        for (bench, metric), value in sorted(latency_data.items()):
            lines.append(f"| {bench} | {metric} | {value:.1f} |")
        lines.append("")
    else:
        lines.append("\n## Latency\n")
        lines.append("*Data not found (latency.csv)*\n")

    # First-vs-second link ordering results
    fvs_file = results_dir / "first_vs_second.csv"
    if fvs_file.exists():
        fvs_data = parse_first_vs_second_csv(fvs_file)
        plot_file = plot_first_vs_second(fvs_data, str(output_dir))

        lines.append("\n## nvJitLink Init Cost: First vs Second Link in Process\n")
        if plot_file:
            lines.append(f"![first vs second]({plot_file})\n")

    report = "\n".join(lines)

    with open(output_file, "w") as f:
        f.write(report)

    print(f"Report written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze AOT benchmark results")
    parser.add_argument(
        "--results-dir", default="results", help="Directory with benchmark outputs"
    )
    parser.add_argument(
        "--output", default="results/report.md", help="Output Markdown report file"
    )
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
