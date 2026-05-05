"""
Experiment Runner: Direction Bias in Forward vs Bidirectional Trajectories

실험 설계:
  - Condition A: BaselineKnowAgent (정방향만)
  - Condition B: BidirectionalKnowAgent (역방향 힌트 + 정방향 실행)
  - Metrics (core): SFR, FAE, EM/F1 (HotpotQA) / SR (ALFWorld)
  - Metrics (supplementary): PAE

벤치마크:
  - HotpotQA: easy / medium / hard (각 n=50, 총 150)
  - ALFWorld: 6 task types (총 134 환경)

실행 예시:
  # HotpotQA
  python3 -m experiments.run_experiment --benchmark hotpotqa --llm gpt-4.1-mini --split easy --n 50

  # ALFWorld
  python3 -m experiments.run_experiment --benchmark alfworld --llm gpt-4.1-mini

  # Mock LLM (API 없이 테스트)
  python3 -m experiments.run_experiment --mock --n 20

  # 기존 파일만 분석
  python3 -m experiments.run_experiment --analyze-only \
    --baseline-file results/baseline_hotpotqa_easy.jsonl \
    --bidir-file    results/bidirectional_hotpotqa_easy.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Path_Generation"))

import joblib

from experiments.bias_metrics import (
    compare_agents,
    load_results_from_jsonl,
    parse_action_sequence_from_prompt,
    print_comparison_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# 결과 로깅
# ─────────────────────────────────────────────────────────────────────────────

def log_agent_result(agent, file_path: str, agent_type: str) -> None:
    """에이전트 실행 결과를 JSONL에 저장한다."""
    record = {
        "question": agent.question,
        "answer": agent.key,
        "correct": agent.is_correct(),
        "reward": agent.reward()[0],
        "halted": agent.is_halted(),
        "error": agent.run_error,
        "agent_type": agent_type,
        "action_sequence": getattr(agent, "action_history", []),
        "prompt": agent._build_agent_prompt(),
    }
    if hasattr(agent, "backward_plan"):
        record["backward_plan"] = agent.backward_plan
        record["backward_plan_generated"] = agent.backward_plan_generated

    with open(file_path, "a") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def log_alfworld_result(
    name: str,
    reward: float,
    traj: str,
    backward_plan: str,
    agent_type: str,
    file_path: str,
) -> None:
    """ALFWorld 실행 결과를 JSONL에 저장한다."""
    # action sequence: '> action' 라인에서 파싱
    action_sequence = [
        line[2:].strip()
        for line in traj.split("\n")
        if line.startswith("> ") and not line.startswith("> ActionPath")
        and not line.startswith("> Think")
    ]
    record = {
        "name": name,
        "result": bool(reward),
        "reward": float(reward),
        "correct": bool(reward),
        "traj": traj,
        "agent_type": agent_type,
        "action_sequence": action_sequence,
        "backward_plan": backward_plan,
    }
    with open(file_path, "a") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로더
# ─────────────────────────────────────────────────────────────────────────────

def _install_pandas_compat_shim() -> None:
    """Legacy pickle compat: pandas<2.0의 pandas.core.indexes.numeric 경로 복원."""
    import sys, types
    if "pandas.core.indexes.numeric" in sys.modules:
        return
    try:
        import pandas.core.indexes  # noqa: F401
        from pandas import Index as _Idx
        shim = types.ModuleType("pandas.core.indexes.numeric")
        for _name in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
            setattr(shim, _name, _Idx)
        sys.modules["pandas.core.indexes.numeric"] = shim
    except Exception:
        pass


def load_hotpotqa_data(split: str = "easy", n: int = 50) -> list[tuple[str, str]]:
    """HotpotQA test split을 로드한다."""
    _install_pandas_compat_shim()
    data_dir = Path(__file__).parent.parent / "Path_Generation" / "hotpotqa_run" / "data" / "test"
    file_path = data_dir / f"{split}.joblib"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = joblib.load(str(file_path))
    qa_pairs = []

    try:
        import pandas as pd
        is_df = isinstance(data, pd.DataFrame)
    except ImportError:
        is_df = False

    if is_df:
        iterator = (row.to_dict() for _, row in data.head(n).iterrows())
    else:
        iterator = iter(data[:n])

    for item in iterator:
        if isinstance(item, dict):
            question = item.get("question", "")
            answer = item.get("answer", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            question, answer = item[0], item[1]
        else:
            continue
        if question and answer:
            qa_pairs.append((question, answer))

    print(f"Loaded {len(qa_pairs)} QA pairs from HotpotQA [{split}]")
    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# HotpotQA 실험
# ─────────────────────────────────────────────────────────────────────────────

def run_hotpotqa_baseline(
    qa_pairs: list[tuple[str, str]], llm, output_file: str, resume: bool = True,
    force_retrieve_first: bool = False,
) -> None:
    from experiments.bidirectional_agent import BaselineKnowAgent

    done = _load_done(output_file) if resume else set()
    label = "Retrieve-oracle" if force_retrieve_first else "Baseline"
    print(f"[HotpotQA {label}] {len(done)} already done, running {len(qa_pairs)-len(done)} more")

    for i, (question, answer) in enumerate(qa_pairs):
        if question in done:
            continue
        print(f"\n[{i+1}/{len(qa_pairs)}] {question[:80]}...")
        agent = BaselineKnowAgent(question=question, key=answer, llm=llm,
                                  force_retrieve_first=force_retrieve_first)
        try:
            agent.run()
        except Exception as e:
            print(f"  Error: {e}")
            agent.run_error = True
        log_agent_result(agent, output_file, "baseline")
        print(f"  EM={agent.is_correct()}, F1={agent.reward()[0]:.3f}, seq={agent.action_history}")


def run_hotpotqa_bidirectional(
    qa_pairs: list[tuple[str, str]],
    llm,
    output_file: str,
    backward_llm=None,
    resume: bool = True,
    bp_mode: str = "rigid",
    freedom_clause: bool = False,
) -> None:
    from experiments.bidirectional_agent import BidirectionalKnowAgent

    done = _load_done(output_file) if resume else set()
    print(f"[HotpotQA Bidirectional bp_mode={bp_mode} freedom={freedom_clause}] {len(done)} already done, running {len(qa_pairs)-len(done)} more")

    for i, (question, answer) in enumerate(qa_pairs):
        if question in done:
            continue
        print(f"\n[{i+1}/{len(qa_pairs)}] {question[:80]}...")
        agent = BidirectionalKnowAgent(
            question=question, key=answer, llm=llm, backward_llm=backward_llm,
            bp_mode=bp_mode, freedom_clause=freedom_clause,
        )
        try:
            agent.run()
        except Exception as e:
            print(f"  Error: {e}")
            agent.run_error = True
        log_agent_result(agent, output_file, "bidirectional")
        print(f"  EM={agent.is_correct()}, F1={agent.reward()[0]:.3f}, seq={agent.action_history}")
        if agent.backward_plan_generated:
            print(f"  Suggested 1st: {agent.backward_agent.get_suggested_first_action()}")


def run_hotpotqa_backward_only(
    qa_pairs: list[tuple[str, str]],
    llm,
    output_file: str,
    backward_llm=None,
    resume: bool = True,
) -> None:
    """B 조건: 역방향 계획만 생성 후 그 시퀀스를 고정 실행 (LLM 재추론 없음)."""
    from experiments.bidirectional_agent import BackwardOnlyAgent

    done = _load_done(output_file) if resume else set()
    print(f"[HotpotQA BackwardOnly] {len(done)} already done, running {len(qa_pairs)-len(done)} more")

    for i, (question, answer) in enumerate(qa_pairs):
        if question in done:
            continue
        print(f"\n[{i+1}/{len(qa_pairs)}] {question[:80]}...")
        agent = BackwardOnlyAgent(
            question=question, key=answer, llm=llm, backward_llm=backward_llm
        )
        try:
            agent.run()
        except Exception as e:
            print(f"  Error: {e}")
            agent.run_error = True
        log_agent_result(agent, output_file, "backward_only")
        print(f"  EM={agent.is_correct()}, F1={agent.reward()[0]:.3f}, seq={agent.action_history}")


# ─────────────────────────────────────────────────────────────────────────────
# ALFWorld 실험
# ─────────────────────────────────────────────────────────────────────────────

ALF_PREFIXES = {
    "pick_and_place":        "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place":  "heat",
    "pick_cool_then_place":  "cool",
    "look_at_obj":           "examine",
    "pick_two_obj":          "puttwo",
}


def _build_alfworld_llm_fn(llm):
    """ALFWorld 실행 루프에서 사용하는 단순 LLM 함수."""
    def call(prompt: str) -> str:
        try:
            return llm(prompt, stop=["\n"], max_tokens=128)
        except Exception:
            return ""
    return call


def _run_alfworld_episode(
    env,
    prompt: str,
    llm_fn,
    token_enc,
    max_steps: int = 50,
    context_limit: int = 4000,
) -> tuple[float, str]:
    """하나의 ALFWorld 에피소드를 실행한다."""
    def process_ob(ob: str) -> str:
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2:]
        return ob

    ob, info = env.reset()
    ob = "\n".join(ob[0].split("\n\n")[1:])
    init_prompt = prompt + ob + "\n>"
    running_prompt = ""
    traj = ob + "\n"

    try:
        action = llm_fn(init_prompt + running_prompt)
        action = action.strip("\n").strip().replace("\n", "")
        observation, reward, done, info = env.step([action])
        observation = process_ob(observation[0])
        reward, done = info["won"][0], done[0]
        traj += f"> {action}\n"
        running_prompt += f" {action}\n>"

        for step in range(1, max_steps):
            if len(token_enc.encode(init_prompt + running_prompt)) > context_limit:
                break
            action = llm_fn(init_prompt + running_prompt)
            action = action.strip("\n").strip().replace("\n", "")
            observation, reward, done, info = env.step([action])
            observation = process_ob(observation[0])
            reward, done = info["won"][0], done[0]

            if action.startswith("ActionPath:") or action.startswith("Think:"):
                traj += f"> {action}\n"
                running_prompt += f" {action}\n>"
            else:
                traj += f"> {action}\n{observation}\n"
                running_prompt += f" {action}\n{observation}\n>"
            if done:
                return float(reward), traj

    except Exception as e:
        print(f"  ALFWorld error: {e}")

    return 0.0, traj


def run_alfworld_experiment(
    llm,
    output_file: str,
    agent_type: str,  # "baseline" or "bidirectional"
    backward_llm=None,
    mode: str = "test",
    resume: bool = True,
) -> None:
    """
    ALFWorld 실험을 실행한다.
    agent_type에 따라 역방향 계획 삽입 여부가 달라진다.
    """
    try:
        import alfworld
        import alfworld.agents.environment
        import yaml
        import tiktoken
        token_enc = tiktoken.get_encoding("cl100k_base")
    except ImportError as e:
        print(f"ALFWorld not installed: {e}. Skipping ALFWorld experiment.")
        return

    config_path = Path(__file__).parent.parent / "Path_Generation" / "alfworld_run" / "base_config.yaml"
    sys.path.insert(0, str(Path(__file__).parent.parent / "Path_Generation" / "alfworld_run"))

    with open(config_path) as f:
        config = yaml.safe_load(f)

    from prompts.taskprompt import (
        alf_put_prompt, alf_clean_prompt, alf_heat_prompt,
        alf_cool_prompt, alf_examine_prompt, alf_puttwo_prompt,
    )
    base_prompts = {
        "put": alf_put_prompt, "clean": alf_clean_prompt,
        "heat": alf_heat_prompt, "cool": alf_cool_prompt,
        "examine": alf_examine_prompt, "puttwo": alf_puttwo_prompt,
    }

    split = "eval_out_of_distribution" if mode == "test" else "train"
    game_num = 134 if mode == "test" else 510

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    # Resume 지원
    done_names = set()
    if resume and Path(output_file).exists():
        existing = load_results_from_jsonl(output_file)
        done_names = {r.get("name", "") for r in existing}
        print(f"[ALFWorld {agent_type}] {len(done_names)} already done")

    llm_fn = _build_alfworld_llm_fn(llm)

    # 역방향 계획 생성기 초기화
    backward_agent = None
    if agent_type == "bidirectional":
        from experiments.backward_agent import BackwardPlanningAgent
        from experiments.alfworld_backward_prompts import alfworld_backward_prompt

        class ALFBackwardAgent(BackwardPlanningAgent):
            def generate_backward_plan(self, task_description: str) -> str:
                prompt = alfworld_backward_prompt.format(task_description=task_description)
                try:
                    plan = (backward_llm or llm)(prompt, stop=[], max_tokens=512)
                except TypeError:
                    plan = (backward_llm or llm)(prompt)
                self.last_backward_plan = str(plan).strip()
                self.last_reversed_sequence = self._parse_reversed_sequence(self.last_backward_plan)
                return self.last_backward_plan

        backward_agent = ALFBackwardAgent(backward_llm or llm)

    cnts = [0] * 6
    rs   = [0] * 6

    for game_idx in range(game_num):
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        if name in done_names:
            continue

        print(f"\n[ALFWorld {agent_type} {game_idx+1}/{game_num}] {name}")

        for i, (task_key, prompt_key) in enumerate(ALF_PREFIXES.items()):
            if not name.startswith(task_key):
                continue

            base_prompt = base_prompts[prompt_key]
            backward_plan = ""

            # 역방향 계획 생성 (bidirectional 조건만)
            if agent_type == "bidirectional" and backward_agent:
                # 태스크 설명 추출: ob의 첫 줄 (예: "put some peppershaker on diningtable.")
                task_desc = ob.split("\n")[0] if ob else name
                try:
                    backward_plan = backward_agent.generate_backward_plan(task_desc)
                    # 역방향 힌트를 프롬프트에 삽입
                    hint_block = (
                        f"\n[BACKWARD PLANNING HINT]\n{backward_plan}\n[END BACKWARD PLANNING HINT]\n"
                    )
                    # base_prompt의 마지막 예시 이후, "Here is the task." 직전에 삽입
                    if "Here is the task." in base_prompt:
                        prompt = base_prompt.replace(
                            "Here is the task.",
                            hint_block + "Here is the task.",
                        )
                    else:
                        prompt = base_prompt + hint_block
                except Exception as e:
                    print(f"  Backward plan failed: {e}")
                    prompt = base_prompt
            else:
                prompt = base_prompt

            # 에피소드 실행
            reward, traj = _run_alfworld_episode(
                env, prompt, llm_fn,
                token_enc=token_enc,
            )

            rs[i] += reward
            cnts[i] += 1

            log_alfworld_result(name, reward, traj, backward_plan, agent_type, output_file)
            total = sum(rs)
            total_cnt = sum(cnts)
            sr = total / total_cnt if total_cnt > 0 else 0.0
            print(f"  reward={reward:.0f}, running SR={sr:.3f} ({total_cnt} tasks)")
            break

    print(f"\n[ALFWorld {agent_type}] Final SR: {sum(rs)/sum(cnts):.4f}")
    task_types = list(ALF_PREFIXES.keys())
    for i, (tt, cnt, r) in enumerate(zip(task_types, cnts, rs)):
        if cnt > 0:
            print(f"  {tt:<30} SR={r/cnt:.3f} ({cnt} tasks)")


# ─────────────────────────────────────────────────────────────────────────────
# 분석 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _load_done(file_path: str) -> set[str]:
    if not Path(file_path).exists():
        return set()
    records = load_results_from_jsonl(file_path)
    return {r.get("question", r.get("name", "")) for r in records}


def analyze_existing_results(
    baseline_file: str,
    bidir_file: str,
    output_json: str | None = None,
) -> dict:
    """기존 JSONL 파일로 bias 지표를 분석한다."""
    print(f"Loading: {baseline_file}")
    baseline_records = load_results_from_jsonl(baseline_file)
    bidir_records    = load_results_from_jsonl(bidir_file)

    def extract_sequences(records):
        seqs = []
        for r in records:
            if r.get("action_sequence"):
                seqs.append(r["action_sequence"])
            elif "prompt" in r:
                seq = parse_action_sequence_from_prompt(r["prompt"])
                if seq:
                    seqs.append(seq)
        return seqs

    baseline_seqs = extract_sequences(baseline_records)
    bidir_seqs    = extract_sequences(bidir_records)

    print(f"Baseline: {len(baseline_records)} records, {len(baseline_seqs)} sequences")
    print(f"Bidirectional: {len(bidir_records)} records, {len(bidir_seqs)} sequences")

    comparison = compare_agents(baseline_seqs, bidir_seqs, baseline_records, bidir_records)
    print_comparison_report(comparison)

    if output_json:
        def _conv(obj):
            if hasattr(obj, "item"):   return obj.item()
            if isinstance(obj, dict):  return {k: _conv(v) for k, v in obj.items()}
            if isinstance(obj, list):  return [_conv(v) for v in obj]
            return obj
        with open(output_json, "w") as f:
            json.dump(_conv(comparison), f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_json}")

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direction bias experiment: HotpotQA + ALFWorld"
    )
    parser.add_argument("--benchmark", default="hotpotqa",
                        choices=["hotpotqa", "alfworld", "2wikimultihopqa", "strategyqa"],
                        help="Benchmark to run")
    parser.add_argument("--llm", default="gpt-4.1-mini")
    parser.add_argument("--backward-llm", default=None)
    # HotpotQA / 2WikiMultihopQA / StrategyQA options
    parser.add_argument("--split", default="easy",
                        choices=["easy", "medium", "hard", "validation", "test", "train"])
    parser.add_argument("--n", type=int, default=50)
    # ALFWorld options
    parser.add_argument("--alf-mode", default="test", choices=["test", "train"])
    # Output
    parser.add_argument("--baseline-file", default=None,
                        help="Override output file for baseline (auto-named if not set)")
    parser.add_argument("--bidir-file", default=None,
                        help="Override output file for bidirectional")
    parser.add_argument("--output-json", default=None)
    # Modes
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--bp-mode", default="rigid",
        choices=["rigid", "goal_directed", "minimal", "action_lift"],
        help="Backward plan format: "
             "rigid (Lv.1, reversed sequence with args) | "
             "goal_directed (v2, BR+FP) | "
             "minimal (Option A, 3-line strategic prior) | "
             "action_lift (Lv.2, lifted backward chaining)",
    )
    parser.add_argument(
        "--freedom-clause", action="store_true",
        help="Forward agent prompt에 'plan은 advisory; observation에 답 있으면 Finish 즉시' 명시",
    )
    parser.add_argument(
        "--skip-backonly", action="store_true",
        help="Skip BackwardOnly condition (only run Baseline + JANUS)",
    )
    parser.add_argument(
        "--skip-bidir", action="store_true",
        help="Skip JANUS condition (run only Baseline). 보통 oracle ablation에 사용.",
    )
    parser.add_argument(
        "--force-retrieve-first", action="store_true",
        help="Baseline에서 첫 action이 Search이면 Retrieve로 강제 (oracle ablation)",
    )
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)

    # ── 자동 파일명 설정 ───────────────────────────────────────────────────
    if args.benchmark == "alfworld":
        tag = f"alfworld_{args.alf_mode}"
    else:
        tag = f"{args.benchmark}_{args.split}"

    baseline_file = args.baseline_file or f"experiments/results/baseline_{tag}.jsonl"
    bidir_file    = args.bidir_file    or f"experiments/results/bidirectional_{tag}.jsonl"
    backonly_file = f"experiments/results/backonly_{tag}.jsonl"
    output_json   = args.output_json   or f"experiments/results/comparison_{tag}.json"

    # ── 분석 전용 모드 ─────────────────────────────────────────────────────
    if args.analyze_only:
        analyze_existing_results(baseline_file, bidir_file, output_json)
        return

    # ── LLM 설정 ──────────────────────────────────────────────────────────
    if args.mock:
        from experiments.mock_llm import MockLLM
        llm          = MockLLM(mode="baseline")
        backward_llm = MockLLM(mode="backward")
        print("Using Mock LLM (no API calls)")
    else:
        from hotpotqa_run.llms import get_llm_backend

        class LLMWrapper:
            def __init__(self, backend):
                self.backend = backend
            def __call__(self, prompt, stop=None, max_tokens=128):
                # Caller가 명시적으로 stop=None을 보내면 multi-line 응답을 의미.
                # 빈 list ([])도 동일하게 stop 없음으로 해석.
                # stop이 미지정(default None)인 경우만 step-line cutoff (["\n"]) 적용.
                if stop is None:
                    effective_stop = ["\n"]
                elif stop == [] or stop == "" or stop is False:
                    effective_stop = None
                else:
                    effective_stop = stop
                return self.backend.run(prompt, stop=effective_stop, max_tokens=max_tokens)

        llm          = LLMWrapper(get_llm_backend(args.llm))
        backward_llm = LLMWrapper(get_llm_backend(args.backward_llm or args.llm))

    resume = not args.no_resume

    # ── QA 벤치마크 공통 실행 함수 (3-way ablation) ───────────────────────
    def _run_qa_benchmark(qa_pairs):
        baseline_label = "Retrieve-oracle" if args.force_retrieve_first else "Baseline"
        print("\n" + "="*50)
        print(f"EXPERIMENT A: {baseline_label} KnowAgent  [{tag}]")
        print("="*50)
        run_hotpotqa_baseline(qa_pairs, llm, baseline_file, resume,
                              force_retrieve_first=args.force_retrieve_first)

        if not args.skip_backonly:
            print("\n" + "="*50)
            print(f"EXPERIMENT B: BackwardOnly  [{tag}]")
            print("="*50)
            run_hotpotqa_backward_only(qa_pairs, llm, backonly_file, backward_llm, resume)
        else:
            print(f"\n[skip] EXPERIMENT B (BackwardOnly) skipped via --skip-backonly")

        if not args.skip_bidir:
            print("\n" + "="*50)
            print(f"EXPERIMENT C: Bidirectional (JANUS) [bp_mode={args.bp_mode} freedom={args.freedom_clause}]  [{tag}]")
            print("="*50)
            run_hotpotqa_bidirectional(qa_pairs, llm, bidir_file, backward_llm, resume,
                                       bp_mode=args.bp_mode,
                                       freedom_clause=args.freedom_clause)
        else:
            print(f"\n[skip] EXPERIMENT C (JANUS) skipped via --skip-bidir")

    # ── HotpotQA ──────────────────────────────────────────────────────────
    if args.benchmark == "hotpotqa":
        qa_pairs = load_hotpotqa_data(split=args.split, n=args.n)

        _run_qa_benchmark(qa_pairs)

    # ── 2WikiMultihopQA ───────────────────────────────────────────────────
    elif args.benchmark == "2wikimultihopqa":
        from experiments.benchmark_loaders import load_benchmark
        split = args.split if args.split in ("train", "validation", "test") else "validation"
        qa_pairs = load_benchmark("2wikimultihopqa", split=split, n=args.n)
        _run_qa_benchmark(qa_pairs)

    # ── StrategyQA ────────────────────────────────────────────────────────
    elif args.benchmark == "strategyqa":
        from experiments.benchmark_loaders import load_benchmark
        split = args.split if args.split in ("train", "test") else "train"
        qa_pairs = load_benchmark("strategyqa", split=split, n=args.n)
        _run_qa_benchmark(qa_pairs)

    # ── ALFWorld ──────────────────────────────────────────────────────────
    elif args.benchmark == "alfworld":
        print("\n" + "="*50)
        print(f"EXPERIMENT A: Baseline KnowAgent  [ALFWorld {args.alf_mode}]")
        print("="*50)
        run_alfworld_experiment(
            llm, baseline_file, agent_type="baseline",
            mode=args.alf_mode, resume=resume,
        )

        print("\n" + "="*50)
        print(f"EXPERIMENT B: Bidirectional  [ALFWorld {args.alf_mode}]")
        print("="*50)
        run_alfworld_experiment(
            llm, bidir_file, agent_type="bidirectional",
            backward_llm=backward_llm,
            mode=args.alf_mode, resume=resume,
        )

    # ── 분석 ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    analyze_existing_results(baseline_file, bidir_file, output_json)


if __name__ == "__main__":
    main()
