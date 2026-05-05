"""
Result Analysis & Visualization

논문 Figure 구성:
  [Main]
    Fig 1. First-Action Distribution (SFR 포함) — Baseline vs Bidirectional bar chart
    Fig 2. FAE Comparison — 엔트로피 vs 이론 최대값 bar chart
    Fig 3. Goal Achievement Rate (EM, F1) — Baseline vs Bidirectional grouped bar

  [Supplementary / Appendix]
    Fig S1. Position-wise Entropy (PAE) — bias가 앞 step에 집중되는지 확인

실행:
  python3 -m experiments.analyze_results \
    --baseline-file experiments/results/baseline.jsonl \
    --bidir-file    experiments/results/bidirectional.jsonl \
    --output-dir    experiments/results/figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Path_Generation"))

from experiments.bias_metrics import (
    compare_agents,
    compute_supplementary_metrics,
    load_results_from_jsonl,
    parse_action_sequence_from_prompt,
    print_comparison_report,
    first_action_distribution,
    positionwise_entropy,
)


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
        return plt
    except ImportError:
        return None


COLORS = {"Baseline": "#4C72B0", "Bidirectional": "#DD8452"}


# ─────────────────────────────────────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_sequences(records: list[dict]) -> list[list[str]]:
    sequences = []
    for r in records:
        if r.get("action_sequence"):
            sequences.append(r["action_sequence"])
        elif "prompt" in r:
            seq = parse_action_sequence_from_prompt(r["prompt"])
            if seq:
                sequences.append(seq)
    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1. First-Action Distribution (SFR 포함)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig1_first_action_dist(baseline_seqs, bidir_seqs, output_dir: str, plt) -> None:
    """
    [Main Fig 1]
    Baseline vs Bidirectional의 첫 번째 action 확률 분포 비교.
    SFR 차이가 눈에 보이는 핵심 그림.
    """
    actions = ["Search", "Retrieve", "Lookup", "Finish"]
    b_dist = first_action_distribution(baseline_seqs)
    d_dist = first_action_distribution(bidir_seqs)

    b_vals = [b_dist.get(a, 0) for a in actions]
    d_vals = [d_dist.get(a, 0) for a in actions]

    x = np.arange(len(actions))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b = ax.bar(x - w/2, b_vals, w, label="Baseline",      color=COLORS["Baseline"],      alpha=0.85)
    bars_d = ax.bar(x + w/2, d_vals, w, label="Bidirectional", color=COLORS["Bidirectional"], alpha=0.85)

    for bars in [bars_b, bars_d]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # SFR 강조 화살표
    b_sfr = b_dist.get("Search", 0)
    d_sfr = d_dist.get("Search", 0)
    ax.annotate(
        f"SFR: {b_sfr:.2f}→{d_sfr:.2f}\n(Δ{d_sfr-b_sfr:+.2f})",
        xy=(0 + w/2, max(b_sfr, d_sfr) + 0.04),
        fontsize=9, color="darkred",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="darkred", alpha=0.8),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.set_title("Fig 1.  First-Action Distribution\n(Direction Bias: RQ2 Evidence)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(output_dir, f"fig1_first_action_dist.{ext}"))
    plt.close()
    print("Saved: fig1_first_action_dist")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2. FAE Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig2_fae(comparison: dict, output_dir: str, plt) -> None:
    """
    [Main Fig 2]
    FAE (First-Action Entropy)를 이론 최대값(2.0 bits)과 함께 비교.
    Baseline이 최대값보다 얼마나 낮은지 = bias 정도를 시각화.
    """
    MAX_ENTROPY = 2.0
    b_fae = comparison["baseline"]["fae"]
    d_fae = comparison["bidirectional"]["fae"]

    labels = ["Baseline", "Bidirectional", "Max Possible\n(uniform)"]
    values = [b_fae, d_fae, MAX_ENTROPY]
    colors = [COLORS["Baseline"], COLORS["Bidirectional"], "#999999"]
    alphas = [0.85, 0.85, 0.5]
    hatches = ["", "", "//"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    for bar, hatch, alpha in zip(bars, hatches, alphas):
        bar.set_hatch(hatch)
        bar.set_alpha(alpha)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # bias gap 표시
    ax.annotate("", xy=(0, b_fae), xytext=(0, MAX_ENTROPY),
                arrowprops=dict(arrowstyle="<->", color="darkred", lw=2))
    ax.text(0.25, (b_fae + MAX_ENTROPY)/2,
            f"Bias gap\n{MAX_ENTROPY - b_fae:.3f} bits",
            color="darkred", fontsize=9)

    ax.set_ylabel("Entropy (bits)")
    ax.set_ylim(0, 2.4)
    ax.set_title("Fig 2.  First-Action Entropy (FAE)\n(lower = stronger direction bias)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(output_dir, f"fig2_fae_comparison.{ext}"))
    plt.close()
    print("Saved: fig2_fae_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3. Goal Achievement Rate (GAR)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig3_gar(comparison: dict, output_dir: str, plt) -> None:
    """
    [Main Fig 3]
    EM + F1 성능 비교. RQ1의 핵심 근거.
    bias 지표 개선(Fig 1,2) + 성능 개선(Fig 3) = RQ1 지지.
    """
    b_ga = comparison["baseline"]["goal_achievement"]
    d_ga = comparison["bidirectional"]["goal_achievement"]

    if not b_ga or not d_ga:
        print("Skipping Fig 3: no GAR data")
        return

    metrics = ["Exact Match (EM)", "Average F1"]
    b_vals = [b_ga["exact_match"], b_ga["avg_f1"]]
    d_vals = [d_ga["exact_match"], d_ga["avg_f1"]]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars_b = ax.bar(x - w/2, b_vals, w, label="Baseline",      color=COLORS["Baseline"],      alpha=0.85)
    bars_d = ax.bar(x + w/2, d_vals, w, label="Bidirectional", color=COLORS["Bidirectional"], alpha=0.85)

    for bars in [bars_b, bars_d]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    d_em = comparison["delta"]["exact_match"]
    d_f1 = comparison["delta"]["avg_f1"]
    if d_em is not None:
        ax.text(0.5, 0.97,
                f"ΔEM={d_em:+.3f}   ΔF1={d_f1:+.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="darkgreen" if d_em > 0 else "darkred",
                bbox=dict(boxstyle="round", facecolor="honeydew", edgecolor="green", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fig 3.  Goal Achievement Rate (GAR)\n(RQ1: Does bias reduction improve performance?)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(output_dir, f"fig3_goal_achievement.{ext}"))
    plt.close()
    print("Saved: fig3_goal_achievement")


# ─────────────────────────────────────────────────────────────────────────────
# Fig S1. Position-wise Entropy (보조)
# ─────────────────────────────────────────────────────────────────────────────

def plot_figs1_pae(baseline_seqs, bidir_seqs, output_dir: str, plt) -> None:
    """
    [Supplementary Fig S1]
    Step 위치별 엔트로피 곡선.
    step 1에서 특히 낮다면 'front-loading bias' 증거.
    """
    b_pae = positionwise_entropy(baseline_seqs)
    d_pae = positionwise_entropy(bidir_seqs)
    positions = sorted(set(b_pae) | set(d_pae))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(positions, [b_pae.get(p, 0) for p in positions],
            "o-", label="Baseline",      color=COLORS["Baseline"],      lw=2, ms=7)
    ax.plot(positions, [d_pae.get(p, 0) for p in positions],
            "s-", label="Bidirectional", color=COLORS["Bidirectional"], lw=2, ms=7)

    ax.axhline(y=2.0, linestyle="--", color="gray", lw=1, label="Max entropy (2.0)")
    ax.set_xlabel("Action Step Position")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Fig S1.  Position-wise Action Entropy (PAE)\n[Supplementary] front-loading bias 분석")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Step {p}" for p in positions])
    ax.set_ylim(0, 2.3)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(output_dir, f"figS1_pae.{ext}"))
    plt.close()
    print("Saved: figS1_pae (Supplementary)")


# ─────────────────────────────────────────────────────────────────────────────
# Text Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(comparison: dict, output_file: str) -> None:
    MAX_ENTROPY = 2.0
    b = comparison["baseline"]
    d = comparison["bidirectional"]
    delta = comparison["delta"]

    lines = [
        "=" * 65,
        "DIRECTION BIAS EXPERIMENT REPORT",
        "=" * 65,
        "",
        "── RQ2: 정방향 trajectory에 direction bias가 존재하는가? ─────────",
        f"  [Baseline] SFR = {b['sfr']:.4f}  "
        f"(Search가 {b['sfr']*100:.1f}%의 경우 첫 action으로 선택됨)",
        f"  [Baseline] FAE = {b['fae']:.4f}  "
        f"(이론 최대 {MAX_ENTROPY:.1f} bits의 {b['fae']/MAX_ENTROPY*100:.1f}% 수준)",
        f"  → {'✓ BIAS DETECTED' if comparison['rq2_evidence'] else '✗ BIAS NOT DETECTED'}"
        f"  (FAE < {MAX_ENTROPY*0.8:.1f} 기준)",
        "",
        "── RQ1: 역방향 보조 입력이 bias를 줄이고 성능을 향상시키는가? ────",
        f"  ΔFAE = {delta['fae']:+.4f}  (양수 = bias 감소)",
        f"  ΔSFR = {delta['sfr']:+.4f}  (음수 = Search-first bias 완화)",
    ]

    if delta["exact_match"] is not None:
        lines += [
            f"  ΔEM  = {delta['exact_match']:+.4f}  (양수 = 성능 향상)",
            f"  ΔF1  = {delta['avg_f1']:+.4f}  (양수 = F1 향상)",
        ]

    lines += [
        f"  → {'✓ RQ1 SUPPORTED' if comparison['rq1_supported'] else '✗ RQ1 NOT SUPPORTED'}"
        f"  (ΔFAE>0 AND ΔEM>0 동시 성립 기준)",
        "",
        "── 수치 요약 표 ───────────────────────────────────────────────────",
        f"  {'지표':<20} {'Baseline':>12} {'Bidirectional':>14} {'Δ':>8}",
        f"  {'-'*56}",
        f"  {'SFR':<20} {b['sfr']:>12.4f} {d['sfr']:>14.4f} {delta['sfr']:>+8.4f}",
        f"  {'FAE (bits)':<20} {b['fae']:>12.4f} {d['fae']:>14.4f} {delta['fae']:>+8.4f}",
    ]

    if comparison["baseline"]["goal_achievement"]:
        b_ga = comparison["baseline"]["goal_achievement"]
        d_ga = comparison["bidirectional"]["goal_achievement"]
        lines += [
            f"  {'EM':<20} {b_ga['exact_match']:>12.4f} {d_ga['exact_match']:>14.4f} {delta['exact_match']:>+8.4f}",
            f"  {'F1':<20} {b_ga['avg_f1']:>12.4f} {d_ga['avg_f1']:>14.4f} {delta['avg_f1']:>+8.4f}",
        ]

    lines += ["", "=" * 65]
    report = "\n".join(lines)
    print(report)
    with open(output_file, "w") as f:
        f.write(report)
    print(f"\nReport saved: {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", default="experiments/results/baseline.jsonl")
    parser.add_argument("--bidir-file",    default="experiments/results/bidirectional.jsonl")
    parser.add_argument("--output-dir",    default="experiments/results/figures")
    parser.add_argument("--no-plots",      action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_records = load_results_from_jsonl(args.baseline_file)
    bidir_records    = load_results_from_jsonl(args.bidir_file)
    baseline_seqs    = extract_sequences(baseline_records)
    bidir_seqs       = extract_sequences(bidir_records)

    print(f"Baseline:      {len(baseline_records)} records, {len(baseline_seqs)} sequences")
    print(f"Bidirectional: {len(bidir_records)} records, {len(bidir_seqs)} sequences")

    comparison = compare_agents(baseline_seqs, bidir_seqs, baseline_records, bidir_records)
    print_comparison_report(comparison)

    generate_report(comparison, os.path.join(args.output_dir, "report.txt"))

    def _to_serializable(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(_to_serializable(comparison), f, indent=2, ensure_ascii=False)

    if not args.no_plots:
        plt = _try_import_matplotlib()
        if plt is None:
            print("matplotlib not available; skipping figures")
        else:
            # [Main] 3개 figure
            plot_fig1_first_action_dist(baseline_seqs, bidir_seqs, args.output_dir, plt)
            plot_fig2_fae(comparison, args.output_dir, plt)
            plot_fig3_gar(comparison, args.output_dir, plt)
            # [Supplementary] 1개 figure
            plot_figs1_pae(baseline_seqs, bidir_seqs, args.output_dir, plt)
            print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
