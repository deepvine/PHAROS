"""
보강 실험 (S1-S3, S5) 구현.
첫 action 선택과 성능 사이의 인과관계를 직접 검증한다.

사용법:
    python3 -m experiments.conditional_analysis

산출물:
    - 콘솔: S1~S3, S5 결과 테이블
    - experiments/results/conditional_analysis.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

QA_BENCHES = [
    ("hotpotqa_easy",   "easy"),
    ("hotpotqa_medium", "medium"),
    ("hotpotqa_hard",   "hard"),
    ("2wikimultihopqa_bridge",     "medium"),
    ("2wikimultihopqa_comparison", "medium"),
    ("strategyqa_train",           "easy"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_action(rec: dict) -> str | None:
    seq = rec.get("action_sequence") or []
    return seq[0] if seq else None


def step_count(rec: dict) -> int:
    return len(rec.get("action_sequence") or [])


# ─────────────────────────────────────────────────────────────────────────────
# S1. Conditional First-Action Performance
# ─────────────────────────────────────────────────────────────────────────────
def s1_conditional_em(records: list[dict]) -> dict:
    """첫 action별 EM 분리 측정."""
    by_first = defaultdict(list)  # first_action → [correct flags]
    for r in records:
        fa = first_action(r)
        if fa in ("Search", "Retrieve"):
            by_first[fa].append(bool(r.get("correct")))

    out = {}
    for fa in ("Search", "Retrieve"):
        flags = by_first[fa]
        n = len(flags)
        em = sum(flags) / n if n else None
        out[fa] = {"n": n, "em": em}

    # gap
    if out["Retrieve"]["em"] is not None and out["Search"]["em"] is not None:
        out["gap_R_minus_S"] = out["Retrieve"]["em"] - out["Search"]["em"]
    else:
        out["gap_R_minus_S"] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# S2. Step Efficiency
# ─────────────────────────────────────────────────────────────────────────────
def s2_step_efficiency(records: list[dict]) -> dict:
    """정답/오답 trajectory의 평균 step 수."""
    correct_steps, failed_steps = [], []
    for r in records:
        s = step_count(r)
        if r.get("correct"):
            correct_steps.append(s)
        else:
            failed_steps.append(s)
    return {
        "avg_steps_correct": (sum(correct_steps) / len(correct_steps))
                              if correct_steps else None,
        "avg_steps_failed": (sum(failed_steps) / len(failed_steps))
                             if failed_steps else None,
        "n_correct": len(correct_steps),
        "n_failed": len(failed_steps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# S3. Lookup Quality (시뮬레이션 한계상 proxy 사용)
# ─────────────────────────────────────────────────────────────────────────────
def s3_lookup_proxy(records: list[dict]) -> dict:
    """첫 action 다음에 Lookup이 등장하는 비율 + 그 trajectory의 EM."""
    after_search = {"with_lookup": 0, "with_lookup_correct": 0, "total": 0}
    after_retrieve = {"with_lookup": 0, "with_lookup_correct": 0, "total": 0}

    for r in records:
        seq = r.get("action_sequence") or []
        if len(seq) < 2:
            continue
        first, second = seq[0], seq[1]
        bucket = (after_search if first == "Search"
                  else after_retrieve if first == "Retrieve" else None)
        if bucket is None:
            continue
        bucket["total"] += 1
        if second == "Lookup":
            bucket["with_lookup"] += 1
            if r.get("correct"):
                bucket["with_lookup_correct"] += 1

    def ratio(b):
        return {
            "lookup_followup_rate": (b["with_lookup"] / b["total"]) if b["total"] else None,
            "lookup_then_correct_rate": (b["with_lookup_correct"] / b["with_lookup"])
                                        if b["with_lookup"] else None,
            "n": b["total"],
        }
    return {"after_search": ratio(after_search), "after_retrieve": ratio(after_retrieve)}


# ─────────────────────────────────────────────────────────────────────────────
# 분석 실행
# ─────────────────────────────────────────────────────────────────────────────
def analyze_benchmark(tag: str, difficulty: str) -> dict:
    a_recs = load_jsonl(RESULTS_DIR / f"forward_{tag}.jsonl")
    c_recs = load_jsonl(RESULTS_DIR / f"bidirectional_{tag}.jsonl")
    return {
        "tag": tag,
        "difficulty": difficulty,
        "n_a": len(a_recs),
        "n_c": len(c_recs),
        "S1_a": s1_conditional_em(a_recs),
        "S1_c": s1_conditional_em(c_recs),
        "S2_a": s2_step_efficiency(a_recs),
        "S2_c": s2_step_efficiency(c_recs),
        "S3_a": s3_lookup_proxy(a_recs),
    }


def fmt(x, dec=3):
    return "—" if x is None else f"{x:.{dec}f}"


def print_s1_table(results: list[dict]) -> None:
    print("\n## S1. Conditional First-Action Performance — A(KnowAgent*) only\n")
    print("| 벤치마크 | n(Search) | EM(Search) | n(Retrieve) | EM(Retrieve) | Gap (R−S) |")
    print("|---------|-----------|------------|-------------|--------------|-----------|")
    for r in results:
        s, t = r["S1_a"]["Search"], r["S1_a"]["Retrieve"]
        gap = r["S1_a"]["gap_R_minus_S"]
        gap_str = ("**" + fmt(gap) + "**") if gap is not None and gap > 0.05 else fmt(gap)
        print(f"| {r['tag']:<28} | {s['n']:>9} | {fmt(s['em']):>10} | "
              f"{t['n']:>11} | {fmt(t['em']):>12} | {gap_str:>9} |")


def print_s2_table(results: list[dict]) -> None:
    print("\n## S2. Step Efficiency — A vs C\n")
    print("| 벤치마크 | A: avg(correct) | C: avg(correct) | A: avg(failed) | C: avg(failed) |")
    print("|---------|-----------------|-----------------|----------------|----------------|")
    for r in results:
        a, c = r["S2_a"], r["S2_c"]
        print(f"| {r['tag']:<28} | {fmt(a['avg_steps_correct']):>15} | {fmt(c['avg_steps_correct']):>15} | "
              f"{fmt(a['avg_steps_failed']):>14} | {fmt(c['avg_steps_failed']):>14} |")


def print_s3_table(results: list[dict]) -> None:
    print("\n## S3. Lookup Followup Quality (proxy) — A only\n")
    print("| 벤치마크 | After Search: Lookup% | After Search: Lookup→EM% | After Retrieve: Lookup% | After Retrieve: Lookup→EM% |")
    print("|---------|-----------------------|--------------------------|-------------------------|----------------------------|")
    for r in results:
        s = r["S3_a"]["after_search"]
        t = r["S3_a"]["after_retrieve"]
        print(f"| {r['tag']:<28} | {fmt(s['lookup_followup_rate']):>22} | "
              f"{fmt(s['lookup_then_correct_rate']):>23} | "
              f"{fmt(t['lookup_followup_rate']):>23} | "
              f"{fmt(t['lookup_then_correct_rate']):>26} |")


def print_s5_table(results: list[dict]) -> None:
    print("\n## S5. Per-Difficulty Bias Cost (Gap = EM(Retrieve) − EM(Search) within A)\n")
    print("| 난이도 | 벤치마크 | Gap (R−S) |")
    print("|--------|---------|-----------|")
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append((r["tag"], r["S1_a"]["gap_R_minus_S"]))
    for diff in ("easy", "medium", "hard"):
        for tag, gap in by_diff[diff]:
            print(f"| {diff:<6} | {tag:<28} | {fmt(gap):>9} |")


def main() -> None:
    results = [analyze_benchmark(tag, diff) for tag, diff in QA_BENCHES]

    print_s1_table(results)
    print_s2_table(results)
    print_s3_table(results)
    print_s5_table(results)

    out = RESULTS_DIR / "conditional_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
