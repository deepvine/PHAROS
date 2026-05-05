# HANDOVER: goal_oriented agent medium-set 재평가

## 배경

HotpotQA **medium split (n=100)** 에서 `goal_oriented` agent (Have/Goal/Gap/Bridge 4-line
Thought 구조)가 baseline (KnowAgent\*) 대비 EM이 **0.62 → 0.55 (−0.07)** 로 낮은 원인을
분석하고, 수정을 완료했다. 이 문서는 재평가 실행을 이어받는 사람을 위한 인수인계다.

---

## 분석 결과 요약

`experiments/results/medium_HAVE_GAP_failure_analysis.xlsx` 참조 (Summary + Case01–14 시트).

baseline은 맞히는데 goal_oriented가 틀리는 케이스 **14건**의 실패 패턴:

| 패턴 | 건수 | 원인 |
|------|------|------|
| Cascade halt (parsing 실패 → self-reinforcing loop) | 5 | stop sequence 오발동으로 빈 응답 → scratchpad에 빈 패턴 축적 → 모델이 패턴 모방 → 10 step halt |
| Literal HAVE (verbatim 인용에 갇혀 합성 추론 차단) | 4 | Have 규칙이 관측에서 직접 인용되지 않은 사실을 차단 |
| Goal minimal-form 규칙 → 정답 절단 | 2 | "Lawrence County" → "Lawrence", "Uniondale, New York" → "Uniondale" |
| HAVE 첫-candidate freezing | 1 | Obs1에 정답(Cheryl Campbell)이 있는데 Peter Wight만 추적 → 8 step halt |
| HAVE verbatim 강제 → 환각된 가짜 인용 | 1 | Retrieve 실패 시 모델이 사전지식을 fake quote로 포장 |
| Literal HAVE → 한 단계 더 필요한데 stop | 1 | formal name(Erwin Rommel)에서 멈춰 popular name(Desert Fox) 미도달 |

---

## 이번에 수정한 내용

### 1. `Path_Generation/hotpotqa_run/agent_arch.py`

**(a) Stop sequence 정리** (`_single_step_generate`, ~L210)

```python
# 수정 전
stop=[f"\nObservation {self.step_n}:", "Observation:"]

# 수정 후
stop=[f"\nObservation {self.step_n}:"]
```

bare `"Observation:"` 를 제거했다. Thought 안에서 `"From Observation 1: ..."` 같은
인용 표현이 나오는 순간 생성이 끊겨 빈 응답이 만들어지는 것이 cascade halt의 1차 원인.

**(b) Action 정규식 완화** (`_single_step_generate`)

```python
# 수정 후: 번호 없는 "Action:" 형식도 fallback 매칭
ac_match = re.search(rf'Action\s*{self.step_n}\s*:\s*([^\n]+)', full) \
        or re.search(r'Action\s*:\s*([^\n]+)', full)
```

**(c) Cascade detection** (`__reset_agent` + `step`)

빈 action이 **2회 연속**이면 즉시 `Finish[information unavailable]`로 강제 종료.
step budget(max=10)을 Invalid Action 반복으로 소진하는 대신 조기 탈출.

### 2. `experiments/prompts/goal_oriented_v2_5_en.txt`

**(a) Goal 섹션에 multi-token suffix 보존 규칙 추가**

`"Lawrence County"` → `"Lawrence"` 절단을 방지하는 명시적 규칙 추가.

**(b) Example 5 추가 (Retrieve 실패 → Search fallback)**

`"Could not find"` 응답 시 fake quote를 생성하지 말고 Search로 전환하는 패턴을
few-shot으로 시연. Fannie Lee Chaney 케이스와 동일 구조.

---

## 다음 할 일: medium 재평가 실행

```bash
cd /workspace/PHAROS

# 기존 결과 백업
cp experiments/results/goal_oriented_hotpotqa_medium.jsonl \
   experiments/results/goal_oriented_hotpotqa_medium_before_fix.jsonl

# 재실행 (n=100, medium, no-resume으로 처음부터)
python3 -m experiments.run_goal_oriented \
  --split medium --n 100 --llm gpt-4.1-mini --no-resume
```

baseline은 수정이 없으므로 `baseline_hotpotqa_medium.jsonl`을 그대로 사용.

실행 후 비교:
```bash
python3 -m experiments.run_goal_oriented --analyze-only --split medium
```

### 기대 결과

| 지표 | 수정 전 | 수정 후 예상 |
|------|---------|------------|
| EM | 0.55 | 0.59–0.62 |
| Halt rate | 17% | 8–10% |
| Finish rate | 83% | 90%+ |

---

## 주요 파일 위치

| 파일 | 설명 |
|------|------|
| `experiments/goal_oriented_agent.py` | GoalOrientedKnowAgent 클래스 |
| `experiments/prompts/goal_oriented_v2_5_en.txt` | Have/Goal/Gap/Bridge 프롬프트 (수정됨) |
| `Path_Generation/hotpotqa_run/agent_arch.py` | BaseAgent / step / _single_step_generate (수정됨) |
| `experiments/run_goal_oriented.py` | 실험 실행 CLI |
| `experiments/results/goal_oriented_hotpotqa_medium.jsonl` | 수정 전 결과 (100건) |
| `experiments/results/medium_HAVE_GAP_failure_analysis.xlsx` | 14건 실패 케이스 상세 분석 |

---

## 미해결 과제 (이후 검토)

- **Cheryl Campbell 류 (candidate freezing)**: 코드/프롬프트 수정으로 아직 안 잡힘.
  → few-shot Example: "Obs1에 후보 N명 → 순차 검증" 패턴 추가 필요 (~600 tok).
  → 단, 현재 preamble이 4138 tok으로 늘었으니 context_len을 7000으로 늘리는 게 선행.
- **Literal HAVE → 합성 추론**: Chang Ucchin, Norbert Holm 류는 프롬프트 규칙 완화가 필요.
  → Gap 판단 기준에 "1-hop 외부 추론 허용" 조건 추가 검토.
- **easy / hard split**: 같은 수정이 easy/hard에서도 유효한지 확인 필요.
