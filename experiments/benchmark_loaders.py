"""
Benchmark Data Loaders

지원 벤치마크:
  1. HotpotQA       - Multi-hop QA (easy/medium/hard)
  2. ALFWorld       - Interactive household tasks (6 types)
  3. 2WikiMultihopQA - Multi-hop QA with bridge/comparison subtypes
  4. StrategyQA     - Yes/No strategic reasoning

모두 (question, answer) 튜플 리스트를 반환하는 통일된 인터페이스를 제공한다.
기존 HotpotQA agent 구조(Search/Retrieve/Lookup/Finish)를 재사용 가능한
QA 형식 벤치마크에 우선 적용한다.

데이터 다운로드:
  # 2WikiMultihopQA
  pip install datasets
  (자동 다운로드 — HuggingFace Hub: "xanhho/2WikiMultihopQA")

  # StrategyQA
  (자동 다운로드 — HuggingFace Hub: "wics/strategy-qa")
"""

import json
import random
from pathlib import Path
from typing import Literal

import joblib


# ─────────────────────────────────────────────────────────────────────────────
# 통합 인터페이스
# ─────────────────────────────────────────────────────────────────────────────

BenchmarkName = Literal["hotpotqa", "2wikimultihopqa", "strategyqa"]


def load_benchmark(
    name: BenchmarkName,
    split: str = "test",
    n: int = 50,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    벤치마크 데이터를 (question, answer) 리스트로 반환한다.

    Args:
        name: 벤치마크 이름
        split: 데이터 split (hotpotqa: easy/medium/hard, 나머지: test/validation)
        n: 샘플 수 (-1이면 전체)
        seed: 랜덤 샘플링 시드

    Returns:
        list of (question, answer) tuples
    """
    loaders = {
        "hotpotqa":        _load_hotpotqa,
        "2wikimultihopqa": _load_2wiki,
        "strategyqa":      _load_strategyqa,
    }

    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Choose from {list(loaders)}")

    pairs = loaders[name](split=split)

    if n > 0 and len(pairs) > n:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, n)

    print(f"[{name}] Loaded {len(pairs)} samples (split={split})")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 1. HotpotQA
# ─────────────────────────────────────────────────────────────────────────────

def _load_hotpotqa(split: str = "easy") -> list[tuple[str, str]]:
    """
    HotpotQA test set 로드.
    split: "easy" | "medium" | "hard"
    """
    data_dir = (
        Path(__file__).parent.parent
        / "Path_Generation" / "hotpotqa_run" / "data" / "test"
    )
    file_path = data_dir / f"{split}.joblib"

    if not file_path.exists():
        raise FileNotFoundError(
            f"HotpotQA [{split}] not found: {file_path}\n"
            "The file should already exist in the repository."
        )

    data = joblib.load(str(file_path))
    pairs = []
    for item in data:
        if isinstance(item, dict):
            q, a = item.get("question", ""), item.get("answer", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            q, a = item[0], item[1]
        else:
            continue
        if q and a:
            pairs.append((q, a))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 2. 2WikiMultihopQA
# ─────────────────────────────────────────────────────────────────────────────

def _load_2wiki(split: str = "validation") -> list[tuple[str, str]]:
    """
    2WikiMultihopQA 로드.

    HuggingFace mirror들이 모두 손상되어(2026-04 기준 script-based blocked /
    parquet 누락) Alab-NII 공식 dropbox 미러에서 직접 받은 로컬 JSON을 사용.
    다운로드 절차: experiments/.cache/2wiki_raw/data.zip 압축 해제

    split: "train" | "validation" (=dev) | "test"
    """
    import json
    raw_dir = Path(__file__).parent / ".cache" / "2wiki_raw"
    file_map = {
        "train":      raw_dir / "train.json",
        "validation": raw_dir / "dev.json",
        "dev":        raw_dir / "dev.json",
        "test":       raw_dir / "test.json",
    }
    file_path = file_map.get(split, raw_dir / "dev.json")
    if not file_path.exists():
        raise FileNotFoundError(
            f"2WikiMultihopQA file not found: {file_path}. "
            "Download from https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip "
            f"and unzip into {raw_dir}"
        )

    with open(file_path) as f:
        data = json.load(f)

    pairs = []
    for item in data:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def _load_2wiki_with_types() -> list[dict]:
    """type 필드까지 포함한 dev set 전체를 dict 리스트로 반환."""
    import json
    raw_dir = Path(__file__).parent / ".cache" / "2wiki_raw"
    file_path = raw_dir / "dev.json"
    if not file_path.exists():
        return []
    with open(file_path) as f:
        return json.load(f)


def get_2wiki_subtype_split(
    pairs: list[tuple[str, str]],
    subtype: Literal["bridge", "comparison", "compositional", "inference"],
    n: int = 50,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    2WikiMultihopQA를 question subtype별로 필터링한다.
    (HuggingFace 데이터셋에 type 필드 있는 경우 활용)
    """
    try:
        from datasets import load_dataset
        cache_dir = Path(__file__).parent / ".cache" / "2wikimultihopqa"
        ds = load_dataset("xanhho/2WikiMultihopQA", split="validation",
                          cache_dir=str(cache_dir), trust_remote_code=True)
        filtered = [
            (item["question"], item["answer"])
            for item in ds
            if item.get("type", "").lower() == subtype
        ]
        rng = random.Random(seed)
        if len(filtered) > n:
            filtered = rng.sample(filtered, n)
        return filtered
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 3. StrategyQA
# ─────────────────────────────────────────────────────────────────────────────

def _load_strategyqa(split: str = "test") -> list[tuple[str, str]]:
    """
    StrategyQA 로드 (HuggingFace datasets).

    특징:
      - 모든 답이 Yes / No (binary)
      - 답하려면 여러 implicit 추론 단계가 필요 ("전략적 추론")
      - Direction Bias 관점에서 흥미로운 점:
        정방향 에이전트가 Yes/No 질문에서도 동일한 Search-first bias를 보이는가?
        → bias가 question type에 무관하게 발생하는 구조적 현상임을 보이는 데 활용

    split: "train" | "test" (StrategyQA에는 validation이 없을 수 있음)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    cache_dir = Path(__file__).parent / ".cache" / "strategyqa"

    # 신규 datasets는 script-based repos 차단 → parquet 기반 mirror 사용.
    repos_to_try = ["ChilleD/StrategyQA", "wics/strategy-qa"]
    ds = None
    last_err = None
    for repo in repos_to_try:
        try:
            ds = load_dataset(repo, split=split, cache_dir=str(cache_dir), trust_remote_code=True)
            break
        except Exception as e:
            last_err = e
            try:
                ds = load_dataset(repo, split="train", cache_dir=str(cache_dir), trust_remote_code=True)
                break
            except Exception as e2:
                last_err = e2
    if ds is None:
        raise RuntimeError(f"Could not load StrategyQA from any mirror: {last_err}")

    pairs = []
    for item in ds:
        question = item.get("question", "").strip()
        # StrategyQA answer: True/False boolean → "yes"/"no" 변환
        raw_answer = item.get("answer", item.get("facts", None))
        if isinstance(raw_answer, bool):
            answer = "yes" if raw_answer else "no"
        elif isinstance(raw_answer, str):
            answer = raw_answer.strip().lower()
            if answer not in ("yes", "no", "true", "false"):
                answer = raw_answer.strip()
        else:
            continue

        if question:
            pairs.append((question, answer))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 벤치마크 메타데이터
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_META = {
    "hotpotqa": {
        "full_name":    "HotpotQA",
        "task_type":    "Multi-hop QA",
        "answer_type":  "span (entity / date / number / yes-no)",
        "action_space": ["Search", "Retrieve", "Lookup", "Finish"],
        "splits":       ["easy", "medium", "hard"],
        "recommended_n": 50,
        "source":       "Yang et al., 2018",
        "bias_focus":   "Search vs Retrieve 선택 편향",
    },
    "2wikimultihopqa": {
        "full_name":    "2WikiMultihopQA",
        "task_type":    "Multi-hop QA (bridge / comparison / compositional / inference)",
        "answer_type":  "span",
        "action_space": ["Search", "Retrieve", "Lookup", "Finish"],
        "splits":       ["validation"],
        "recommended_n": 50,
        "source":       "Ho et al., 2020",
        "bias_focus":   "다양한 추론 유형에서 편향이 동일하게 발생하는지 확인",
    },
    "strategyqa": {
        "full_name":    "StrategyQA",
        "task_type":    "Yes/No Strategic Reasoning",
        "answer_type":  "yes / no (binary)",
        "action_space": ["Search", "Retrieve", "Lookup", "Finish"],
        "splits":       ["train"],
        "recommended_n": 50,
        "source":       "Geva et al., 2021",
        "bias_focus":   "Yes/No 질문에서도 동일한 direction bias가 나타나는가",
    },
}


def print_benchmark_overview() -> None:
    """전체 벤치마크 구성 요약을 출력한다."""
    print("\n" + "=" * 65)
    print("BENCHMARK OVERVIEW")
    print("=" * 65)
    for key, meta in BENCHMARK_META.items():
        print(f"\n[{meta['full_name']}]")
        print(f"  Task:        {meta['task_type']}")
        print(f"  Answer:      {meta['answer_type']}")
        print(f"  Bias Focus:  {meta['bias_focus']}")
        print(f"  Source:      {meta['source']}")
    print("=" * 65 + "\n")
