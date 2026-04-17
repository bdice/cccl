#!/usr/bin/env python3
"""analyze_results.py — Parse NVBench JSON + latency CSV and generate a Markdown report with plots.

Usage:
  python analyze_results.py [--results-dir results/] [--output report.md]
"""

import argparse
import csv
import json
import os
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Reduce latency
    reduce_bars = {
        "CUB median": latency_data.get(("cub_reduce", "launch_median_us"), 0),
        "AOT median": latency_data.get(("aot_reduce", "launch_median_us"), 0),
        "AOT cold link": latency_data.get(("aot_reduce", "cold_link_us"), 0),
        "AOT hot link": latency_data.get(("aot_reduce", "hot_link_us"), 0),
    }
    ax1.barh(
        list(reduce_bars.keys()),
        list(reduce_bars.values()),
        color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"],
    )
    ax1.set_xlabel("Latency (us)")
    ax1.set_title("Reduce Latency")

    # Transform latency
    transform_bars = {
        "CUB median": latency_data.get(("cub_transform", "launch_median_us"), 0),
        "AOT median": latency_data.get(("aot_transform", "launch_median_us"), 0),
        "AOT cold link": latency_data.get(("aot_transform", "cold_link_us"), 0),
        "AOT hot link": latency_data.get(("aot_transform", "hot_link_us"), 0),
    }
    ax2.barh(
        list(transform_bars.keys()),
        list(transform_bars.values()),
        color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"],
    )
    ax2.set_xlabel("Latency (us)")
    ax2.set_title("Transform Latency")

    fig.suptitle("Launch & Link Latency (int32, N=1M)", fontsize=14)
    fig.tight_layout()

    filename = "latency_comparison.png"
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
