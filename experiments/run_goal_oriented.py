"""
Goal-Oriented vs Baseline runner (HotpotQA).

기존 multi-condition runner를 건드리지 않고 baseline ↔ goal-oriented 두 조건만 깔끔히 비교.

사용 예:
  # smoke (3 문항)
  python3 -m experiments.run_goal_oriented --split medium --n 3 --llm gpt-4.1-mini

  # 풀 실행 (50 문항)
  python3 -m experiments.run_goal_oriented --split medium --n 50 --llm gpt-4.1-mini

  # 분석만
  python3 -m experiments.run_goal_oriented --analyze-only --split medium
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Path_Generation"))

from experiments.bias_metrics import (
    compare_agents,
    load_results_from_jsonl,
    parse_action_sequence_from_prompt,
    print_comparison_report,
)
from experiments.run_experiment import (
    load_hotpotqa_data,
    log_agent_result,
    _load_done,
)


def run_goal_oriented(qa_pairs, llm, output_file: str, resume: bool = True) -> None:
    from experiments.goal_oriented_agent import GoalOrientedKnowAgent

    done = _load_done(output_file) if resume else set()
    print(f"[GoalOriented] {len(done)} already done, running {len(qa_pairs)-len(done)} more")

    for i, (question, answer) in enumerate(qa_pairs):
        if question in done:
            continue
        print(f"\n[{i+1}/{len(qa_pairs)}] {question[:80]}...")
        agent = GoalOrientedKnowAgent(question=question, key=answer, llm=llm)
        try:
            agent.run()
        except Exception as e:
            print(f"  Error: {e}")
            agent.run_error = True
        log_agent_result(agent, output_file, "goal_oriented")
        print(f"  EM={agent.is_correct()}, F1={agent.reward()[0]:.3f}, seq={agent.action_history}")


def run_baseline(qa_pairs, llm, output_file: str, resume: bool = True) -> None:
    from experiments.bidirectional_agent import BaselineKnowAgent

    done = _load_done(output_file) if resume else set()
    print(f"[Baseline] {len(done)} already done, running {len(qa_pairs)-len(done)} more")

    for i, (question, answer) in enumerate(qa_pairs):
        if question in done:
            continue
        print(f"\n[{i+1}/{len(qa_pairs)}] {question[:80]}...")
        agent = BaselineKnowAgent(question=question, key=answer, llm=llm, context_len=6000)
        try:
            agent.run()
        except Exception as e:
            print(f"  Error: {e}")
            agent.run_error = True
        log_agent_result(agent, output_file, "baseline")
        print(f"  EM={agent.is_correct()}, F1={agent.reward()[0]:.3f}, seq={agent.action_history}")


def _extract_sequences(records):
    seqs = []
    for r in records:
        if r.get("action_sequence"):
            seqs.append(r["action_sequence"])
        elif "prompt" in r:
            seq = parse_action_sequence_from_prompt(r["prompt"])
            if seq:
                seqs.append(seq)
    return seqs


def _format_compliance(records) -> dict:
    """goal-oriented 출력에서 'Goal:', 'Need:', 'Gap:' 라인이 들어간 비율을 측정."""
    n = len(records)
    if n == 0:
        return {"records": 0}
    goal = need = gap = 0
    for r in records:
        prompt = r.get("prompt", "")
        if "END OF EXAMPLES" in prompt:
            after = prompt.split("END OF EXAMPLES")[-1]
        elif "End of examples" in prompt:
            after = prompt.split("End of examples")[-1]
        else:
            after = prompt
        if "Goal:" in after:
            goal += 1
        if "Need:" in after:
            need += 1
        if "Gap:" in after:
            gap += 1
    return {
        "records": n,
        "Goal_present": goal / n,
        "Need_present": need / n,
        "Gap_present": gap / n,
    }


def analyze(baseline_file: str, goal_file: str, output_json: str | None = None) -> dict:
    print(f"Loading baseline: {baseline_file}")
    baseline_records = load_results_from_jsonl(baseline_file)
    print(f"Loading goal-oriented: {goal_file}")
    goal_records = load_results_from_jsonl(goal_file)

    # paired 비교: 같은 질문만
    baseline_q = {r["question"]: r for r in baseline_records if "question" in r}
    goal_q = {r["question"]: r for r in goal_records if "question" in r}
    common = sorted(set(baseline_q) & set(goal_q))
    print(f"Baseline records: {len(baseline_records)}, Goal records: {len(goal_records)}, "
          f"matched questions: {len(common)}")

    if not common:
        print("No matched questions — analysis aborted.")
        return {}

    paired_baseline = [baseline_q[q] for q in common]
    paired_goal     = [goal_q[q] for q in common]

    baseline_seqs = _extract_sequences(paired_baseline)
    goal_seqs     = _extract_sequences(paired_goal)

    print(f"  baseline sequences: {len(baseline_seqs)}, goal sequences: {len(goal_seqs)}")

    # compare_agents는 (baseline, bidir) 라벨을 쓰지만 수학은 동일.
    comparison = compare_agents(baseline_seqs, goal_seqs, paired_baseline, paired_goal)
    print_comparison_report(comparison)

    # 추가: format compliance (goal-oriented만)
    compliance = _format_compliance(paired_goal)
    print("\n[Format Compliance — goal-oriented]")
    for k, v in compliance.items():
        print(f"  {k}: {v}")

    out = {
        "matched_n": len(common),
        "comparison": comparison,
        "format_compliance": compliance,
    }
    if output_json:
        def _conv(obj):
            if hasattr(obj, "item"): return obj.item()
            if isinstance(obj, dict): return {k: _conv(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_conv(v) for v in obj]
            return obj
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(_conv(out), f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_json}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="gpt-4.1-mini")
    parser.add_argument("--benchmark", default="hotpotqa",
                        choices=["hotpotqa", "2wikimultihopqa", "strategyqa"])
    parser.add_argument("--split", default="medium",
                        help="hotpotqa: easy/medium/hard | 2wiki: validation | strategyqa: train/test")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--run-baseline", action="store_true",
                        help="baseline도 같이 실행 (이미 결과가 있으면 resume)")
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)
    tag = f"{args.benchmark}_{args.split}"
    baseline_file = f"experiments/results/baseline_{tag}.jsonl"
    goal_file     = f"experiments/results/goal_oriented_{tag}.jsonl"
    output_json   = f"experiments/results/comparison_goal_{tag}.json"

    if args.analyze_only:
        analyze(baseline_file, goal_file, output_json)
        return

    # LLM 초기화
    from hotpotqa_run.llms import get_llm_backend

    class LLMWrapper:
        def __init__(self, backend):
            self.backend = backend
        def __call__(self, prompt, stop=None, max_tokens=128):
            if stop is None:
                effective_stop = ["\n"]
            elif stop == [] or stop == "" or stop is False:
                effective_stop = None
            else:
                effective_stop = stop
            return self.backend.run(prompt, stop=effective_stop, max_tokens=max_tokens)

    llm = LLMWrapper(get_llm_backend(args.llm))
    resume = not args.no_resume

    if args.benchmark == "hotpotqa":
        qa_pairs = load_hotpotqa_data(split=args.split, n=args.n)
    else:
        from experiments.benchmark_loaders import load_benchmark
        qa_pairs = load_benchmark(args.benchmark, split=args.split, n=args.n)

    if args.run_baseline:
        print("\n" + "="*50 + f"\nBASELINE  [{tag}]\n" + "="*50)
        run_baseline(qa_pairs, llm, baseline_file, resume)

    print("\n" + "="*50 + f"\nGOAL-ORIENTED  [{tag}]\n" + "="*50)
    run_goal_oriented(qa_pairs, llm, goal_file, resume)

    print("\n" + "="*50 + "\nANALYSIS\n" + "="*50)
    analyze(baseline_file, goal_file, output_json)


if __name__ == "__main__":
    main()
