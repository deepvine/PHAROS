"""
Direction Bias Metrics

실험 목적:
  RQ2: 정방향 trajectory에 direction bias가 존재하는가?
  RQ1: 역방향 보조 입력으로 bias가 줄고 성능이 향상되는가?

────────────────────────────────────────────────────
[핵심 지표] 논문 본문에 사용
  - SFR  (Search-First Rate):     첫 action이 Search인 비율
  - FAE  (First-Action Entropy):  첫 action 분포의 엔트로피
  - GAR  (Goal Achievement Rate): EM + F1 (실제 성능)

[보조 지표] Appendix / 심층 분석용
  - PAE  (Position-wise Entropy): step 위치별 엔트로피
    → "bias가 특히 앞 step에 집중"되는지 보여주는 보완 분석
────────────────────────────────────────────────────
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_action_sequence_from_prompt(prompt_text: str) -> list[str]:
    """
    prompt 텍스트에서 'Action N: ActionType[...]' 패턴을 파싱.
    Examples 영역은 제외하고 실제 실행 부분만 파싱.
    """
    if "END OF EXAMPLES" in prompt_text:
        prompt_text = prompt_text.split("END OF EXAMPLES")[-1]
    pattern = re.compile(r"Action\s+\d+:\s*(\w+)\[")
    return pattern.findall(prompt_text)


def load_results_from_jsonl(file_path: str) -> list[dict]:
    """JSONL 결과 파일을 로드한다."""
    records = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# ① SFR  — Search-First Rate
# ─────────────────────────────────────────────────────────────────────────────

def search_first_rate(sequences: list[list[str]]) -> float:
    """
    첫 번째 action이 'Search'인 비율.

    높을수록 "무조건 Search부터" 편향이 강함.
    Baseline vs Bidirectional의 핵심 비교 지표.
    """
    first_actions = [seq[0] for seq in sequences if seq]
    if not first_actions:
        return 0.0
    return sum(1 for a in first_actions if a == "Search") / len(first_actions)


def first_action_distribution(sequences: list[list[str]]) -> dict[str, float]:
    """첫 action의 확률 분포 반환 (SFR의 전체 버전)."""
    first_actions = [seq[0] for seq in sequences if seq]
    total = len(first_actions)
    if not total:
        return {}
    return {k: v / total for k, v in Counter(first_actions).items()}


# ─────────────────────────────────────────────────────────────────────────────
# ② FAE  — First-Action Entropy
# ─────────────────────────────────────────────────────────────────────────────

def entropy(samples: list[str] | dict[str, float]) -> float:
    """Shannon 엔트로피 H = -Σ p(x) log₂ p(x)."""
    if isinstance(samples, list):
        counts = Counter(samples)
        total = sum(counts.values())
        dist = {k: v / total for k, v in counts.items()}
    else:
        dist = samples

    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def first_action_entropy(sequences: list[list[str]]) -> float:
    """
    첫 번째 action 분포의 엔트로피 (FAE).

    4개 action이 완전히 균등하면 최대 2.0 bits.
    낮을수록 특정 action으로 편향.
    SFR이 편향의 방향을 보여준다면, FAE는 편향의 정도를 수치화.
    """
    first_actions = [seq[0] for seq in sequences if seq]
    return entropy(first_actions)


# ─────────────────────────────────────────────────────────────────────────────
# ③ GAR  — Goal Achievement Rate
# ─────────────────────────────────────────────────────────────────────────────

def goal_achievement_rate(records: list[dict]) -> dict[str, float]:
    """
    실제 성능 지표: EM + F1.

    다양성 지표(SFR, FAE)와 함께 봐야 의미가 있음.
    → bias 감소(↑FAE, ↓SFR) + 성능 향상(↑EM, ↑F1) 이 동시에 성립해야 RQ1 지지.
    """
    if not records:
        return {"exact_match": 0.0, "avg_f1": 0.0, "finish_rate": 0.0, "n": 0}

    em_list   = [r.get("correct", False) for r in records]
    f1_list   = [r.get("reward", 0.0)    for r in records]
    halt_list = [r.get("halted", False)  for r in records]
    finish_count = sum(
        1 for r in records
        if not r.get("halted", False) and not r.get("error", False)
    )

    return {
        "exact_match":  sum(em_list)  / len(em_list),
        "avg_f1":       sum(f1_list)  / len(f1_list),
        "finish_rate":  finish_count  / len(records),
        "halt_rate":    sum(halt_list) / len(records),
        "n": len(records),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ④ PAE  — Position-wise Action Entropy  [보조 지표]
# ─────────────────────────────────────────────────────────────────────────────

def positionwise_entropy(
    sequences: list[list[str]], max_pos: int = 6
) -> dict[int, float]:
    """
    Step 위치별 action 분포의 엔트로피.

    [보조 분석] 앞 step(step 1)에서 엔트로피가 특히 낮다면
    direction bias가 trajectory 초반에 집중됨을 시각적으로 보여줄 수 있음.
    PAE 곡선이 Bidirectional에서 전반적으로 높아지면 bias 완화 증거.
    """
    position_actions: dict[int, list[str]] = defaultdict(list)
    for seq in sequences:
        for i, action in enumerate(seq[:max_pos], start=1):
            position_actions[i].append(action)
    return {pos: entropy(actions) for pos, actions in sorted(position_actions.items())}


# ─────────────────────────────────────────────────────────────────────────────
# 종합 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_core_metrics(
    sequences: list[list[str]],
    records: list[dict] | None = None,
    label: str = "agent",
) -> dict[str, Any]:
    """
    논문 본문용 핵심 3종 지표를 계산한다.

    Returns:
        {
          label, n_trajectories,
          sfr, fae, first_action_distribution,   ← Bias 지표
          goal_achievement,                        ← 성능 지표
        }
    """
    result: dict[str, Any] = {
        "label": label,
        "n_trajectories": len(sequences),
        # ① Bias 지표
        "sfr": search_first_rate(sequences),
        "fae": first_action_entropy(sequences),
        "first_action_distribution": first_action_distribution(sequences),
        # ③ 성능 지표
        "goal_achievement": goal_achievement_rate(records) if records else None,
    }
    return result


def compute_supplementary_metrics(
    sequences: list[list[str]],
    label: str = "agent",
) -> dict[str, Any]:
    """
    Appendix용 보조 지표 (PAE).
    """
    return {
        "label": label,
        "pae": positionwise_entropy(sequences),
    }


def compare_agents(
    baseline_sequences: list[list[str]],
    bidir_sequences:    list[list[str]],
    baseline_records:   list[dict] | None = None,
    bidir_records:      list[dict] | None = None,
) -> dict[str, Any]:
    """
    Baseline vs Bidirectional 비교.

    핵심 판단 로직:
      RQ2: baseline FAE < 이론 최대(2.0) 의 80% → bias 존재
      RQ1: ΔFAE > 0 AND ΔEM > 0 → bias 감소 + 성능 향상 동시 성립
    """
    b = compute_core_metrics(baseline_sequences, baseline_records, label="Baseline")
    d = compute_core_metrics(bidir_sequences,    bidir_records,    label="Bidirectional")

    delta_fae = d["fae"]  - b["fae"]
    delta_sfr = d["sfr"]  - b["sfr"]

    delta_em = delta_f1 = None
    if b["goal_achievement"] and d["goal_achievement"]:
        delta_em = d["goal_achievement"]["exact_match"] - b["goal_achievement"]["exact_match"]
        delta_f1 = d["goal_achievement"]["avg_f1"]       - b["goal_achievement"]["avg_f1"]

    # RQ 판단
    MAX_ENTROPY = 2.0  # log2(4) — 4개 action 균등 분포
    rq2_evidence  = b["fae"] < MAX_ENTROPY * 0.8          # 낮은 엔트로피 = bias 존재
    rq1_supported = (
        delta_fae is not None and delta_fae > 0
        and delta_em is not None and delta_em > 0
    )

    return {
        "baseline":       b,
        "bidirectional":  d,
        "delta": {
            "fae": delta_fae,     # ↑ 양수 = bias 감소
            "sfr": delta_sfr,     # ↓ 음수 = Search-first bias 감소
            "exact_match": delta_em,
            "avg_f1":      delta_f1,
        },
        "rq2_evidence":  rq2_evidence,
        "rq1_supported": rq1_supported,
    }


def print_comparison_report(comparison: dict[str, Any]) -> None:
    """비교 결과를 출력한다."""
    MAX_ENTROPY = 2.0

    print("\n" + "=" * 60)
    print("DIRECTION BIAS ANALYSIS REPORT")
    print("=" * 60)

    for key in ["baseline", "bidirectional"]:
        m = comparison[key]
        ga = m["goal_achievement"] or {}
        print(f"\n[{m['label']}]  n={m['n_trajectories']}")
        print(f"  ── Bias 지표 ──────────────────────────")
        print(f"  SFR (Search-First Rate):  {m['sfr']:.4f}")
        print(f"  FAE (1st-Action Entropy): {m['fae']:.4f}  "
              f"(max={MAX_ENTROPY:.1f}, ratio={m['fae']/MAX_ENTROPY:.2f})")
        print(f"  First-Action Distribution:")
        for action, prob in sorted(m["first_action_distribution"].items()):
            bar = "█" * int(prob * 20)
            print(f"    {action:<10} {prob:.3f}  {bar}")
        if ga:
            print(f"  ── 성능 지표 ──────────────────────────")
            print(f"  EM  (Exact Match):        {ga['exact_match']:.4f}")
            print(f"  F1  (Average F1):         {ga['avg_f1']:.4f}")
            print(f"  Finish Rate:              {ga['finish_rate']:.4f}")

    d = comparison["delta"]
    print(f"\n[Delta: Bidirectional − Baseline]")
    print(f"  ΔFAE  (↑양수 = bias 감소):   {d['fae']:+.4f}")
    print(f"  ΔSFR  (↓음수 = Search bias↓):{d['sfr']:+.4f}")
    if d["exact_match"] is not None:
        print(f"  ΔEM   (↑양수 = 성능 향상):   {d['exact_match']:+.4f}")
        print(f"  ΔF1   (↑양수 = F1 향상):     {d['avg_f1']:+.4f}")

    print(f"\n[Research Question]")
    print(f"  RQ2 (정방향 bias 존재):           "
          f"{'✓ EVIDENCE' if comparison['rq2_evidence'] else '✗ NOT FOUND'}")
    print(f"  RQ1 (역방향 입력 → bias↓ + 성능↑): "
          f"{'✓ SUPPORTED' if comparison['rq1_supported'] else '✗ NOT SUPPORTED'}")
    print("=" * 60 + "\n")
