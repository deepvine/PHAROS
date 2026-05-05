"""
BidirectionalKnowAgent

정방향 KnowAgent에 역방향 계획을 보조 입력으로 결합한 에이전트.

연구 핵심:
  - 기존 KnowAgentHotpotQA: 정방향만 사용 → direction bias 존재 가능
  - BidirectionalKnowAgent: 역방향 계획을 먼저 생성 후 정방향 실행에 활용
    → direction bias 감소, 목표 달성률 향상 여부를 측정

실행 흐름:
  1. BackwardPlanningAgent로 역방향 계획 생성
  2. 역방향 계획을 prompt에 삽입
  3. 기존 KnowAgent와 동일하게 정방향으로 실행
  4. 각 step의 action 선택을 기록 (bias 분석용)
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Path_Generation'))

from hotpotqa_run.agent_arch import BaseAgent, parse_action, format_step
from hotpotqa_run.fewshots import KNOWAGENT_EXAMPLE
from experiments.backward_prompts import bidirectional_prompt, bidirectional_prompt_with_freedom
from experiments.backward_agent import BackwardPlanningAgent


class BidirectionalKnowAgent(BaseAgent):
    """
    역방향 계획을 보조 입력으로 활용하는 bidirectional agent.

    Attributes:
        backward_agent: 역방향 계획 생성기
        backward_plan: 생성된 역방향 계획 텍스트
        action_history: 각 step의 action type 기록 (bias 분석용)
        backward_plan_generated: 역방향 계획 생성 성공 여부
    """

    def __init__(
        self,
        question: str,
        key: str,
        llm,
        backward_llm=None,
        context_len: int = 2000,
        generate_backward: bool = True,
        bp_mode: str = "rigid",
        freedom_clause: bool = False,
    ) -> None:
        """
        Args:
            question: 질문
            key: 정답 (ground truth)
            llm: 정방향 action 생성에 사용할 LLM
            backward_llm: 역방향 계획 생성에 사용할 LLM (None이면 llm과 동일)
            context_len: 최대 컨텍스트 토큰 수
            generate_backward: 역방향 계획 생성 여부 (False면 baseline처럼 동작)
            bp_mode: backward plan 형식
                - "rigid": Lv.1, reversed sequence with concrete arguments
                - "goal_directed": 사용자 의도 v2, backward reasoning + forward plan
                - "minimal": Option A, 3-line strategic prior (no sequencing)
                - "action_lift": Lv.2, lifted backward chaining (action types only)
            freedom_clause: True이면 forward 에이전트 prompt에 "plan은 advisory" 명시
        """
        super().__init__(question, key, llm, context_len)
        self.examples = KNOWAGENT_EXAMPLE
        self.agent_prompt = (
            bidirectional_prompt_with_freedom if freedom_clause else bidirectional_prompt
        )
        self.name = "BidirectionalKnowAgent"
        self.generate_backward = generate_backward
        self.bp_mode = bp_mode
        self.freedom_clause = freedom_clause

        # 역방향 계획 관련
        _backward_llm = backward_llm if backward_llm is not None else llm
        self.backward_agent = BackwardPlanningAgent(_backward_llm, mode=bp_mode)
        self.backward_plan = ""
        self.backward_plan_generated = False

        # Bias 분석을 위한 action 기록
        self.action_history: list[str] = []

        # 에이전트 초기화 시 역방향 계획 생성
        if self.generate_backward:
            self._initialize_backward_plan()

    def _initialize_backward_plan(self) -> None:
        """에이전트 초기화 시 역방향 계획을 한 번 생성한다."""
        try:
            self.backward_plan = self.backward_agent.generate_backward_plan(
                self.question
            )
            self.backward_plan_generated = True
            print(f"[Backward Plan Generated]\n{self.backward_plan}\n")
        except Exception as e:
            print(f"[Backward Plan Failed] {e}")
            self.backward_plan = (
                "Backward planning unavailable. Proceed with forward planning."
            )
            self.backward_plan_generated = False

    def forward(self):
        """KnowAgent와 동일한 forward 흐름 (single-call chat-model 버전)."""
        from hotpotqa_run.agent_arch import parse_action
        _ap, _th, action_text = self._single_step_generate()
        pattern = re.compile(r'\s+(?=\[)')
        action_text = pattern.sub('', action_text)
        action_type, argument = parse_action(action_text)
        self.action_history.append(action_type)
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        """역방향 계획을 포함한 bidirectional prompt를 구성한다."""
        backward_plan_text = (
            self.backward_plan
            if self.backward_plan
            else "No backward plan available. Proceed with forward planning only."
        )
        return self.agent_prompt.format(
            examples=self.examples,
            backward_plan=backward_plan_text,
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def get_first_action(self) -> str:
        """첫 번째로 실행된 action type을 반환한다."""
        return self.action_history[0] if self.action_history else ""

    def get_action_sequence(self) -> list[str]:
        """전체 action 시퀀스를 반환한다."""
        return list(self.action_history)

    def reset_with_new_question(self, question: str, key: str) -> None:
        """새 질문으로 에이전트를 재설정한다."""
        self.question = question
        self.key = key
        self.action_history = []
        self.backward_plan = ""
        self.backward_plan_generated = False
        self.set_qa(question, key)
        if self.generate_backward:
            self._initialize_backward_plan()


class BaselineKnowAgent(BaseAgent):
    """
    역방향 계획 없이 순수 정방향으로만 동작하는 baseline agent.
    BidirectionalKnowAgent와 공정한 비교를 위해 동일한 action_history 기록 기능 포함.
    """

    def __init__(self, question: str, key: str, llm, context_len: int = 2000,
                 force_retrieve_first: bool = False) -> None:
        super().__init__(question, key, llm, context_len)

        from hotpotqa_run.pre_prompt import knowagent_prompt

        self.examples = KNOWAGENT_EXAMPLE
        self.agent_prompt = knowagent_prompt
        self.name = "BaselineKnowAgent"
        self.action_history: list[str] = []
        self.force_retrieve_first = force_retrieve_first

    def forward(self):
        from hotpotqa_run.agent_arch import parse_action
        _ap, _th, action_text = self._single_step_generate()
        pattern = re.compile(r'\s+(?=\[)')
        action_text = pattern.sub('', action_text)
        action_type, argument = parse_action(action_text)

        # Retrieve-oracle ablation: 첫 step에서 Search → Retrieve 강제 (인자 유지).
        # 단, 인자가 비어있는 경우(파싱 실패) 그대로 둠.
        if (
            self.force_retrieve_first
            and self.step_n == 1
            and action_type == "Search"
            and argument
        ):
            action_type = "Retrieve"
            # scratchpad의 마지막 Action 라인을 Retrieve[arg]로 덮어쓰기
            import re as _re
            self.scratchpad = _re.sub(
                r'(Action\s+1:\s*)Search(\[)',
                r'\1Retrieve\2',
                self.scratchpad,
                count=1,
            )

        self.action_history.append(action_type)
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.examples,
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def get_first_action(self) -> str:
        return self.action_history[0] if self.action_history else ""

    def get_action_sequence(self) -> list[str]:
        return list(self.action_history)


class BackwardOnlyAgent(BaseAgent):
    """
    역방향 계획을 생성한 뒤, 그 시퀀스를 그대로 실행하는 에이전트.

    Ablation 목적:
      A (Forward Only) vs B (Backward Only) vs C (Bidirectional) 3-way 비교에서
      B 조건을 담당한다.

    동작 방식:
      1. 역방향 계획 생성: Reversed sequence: Start ← A1 ← A2 ← Finish
      2. 정방향 순서 추출: [A1, A2, Finish]
      3. LLM의 추가 추론 없이 그 순서를 그대로 실행

    한계 (논문 분석 포인트):
      - 역방향 계획이 틀리면 복구 불가 (적응성 부재)
      - C (Bidirectional)가 B보다 강인한 이유를 설명하는 대조군
    """

    def __init__(
        self,
        question: str,
        key: str,
        llm,
        backward_llm=None,
        context_len: int = 2000,
    ) -> None:
        super().__init__(question, key, llm, context_len)
        self.name = "BackwardOnlyAgent"
        self.action_history: list[str] = []

        # 역방향 계획 생성
        _bllm = backward_llm if backward_llm is not None else llm
        self.backward_agent = BackwardPlanningAgent(_bllm)
        self.backward_plan = ""
        self.backward_plan_generated = False
        self._planned_sequence: list[str] = []   # 실행할 action 시퀀스
        self._plan_index: int = 0                # 현재 몇 번째 step인지

        self._initialize_plan()

    def _initialize_plan(self) -> None:
        """역방향 계획에서 실행 시퀀스를 추출한다."""
        try:
            self.backward_plan = self.backward_agent.generate_backward_plan(self.question)
            self.backward_plan_generated = True
            raw_seq = self.backward_agent.last_reversed_sequence  # [A1, A2, ..., Finish[x]]
            # 각 element에서 action type만 추출
            self._planned_sequence = []
            for item in raw_seq:
                action_match = re.match(r'(\w+)\[', item.strip())
                if action_match:
                    self._planned_sequence.append(item.strip())
                elif item.strip().lower() not in ("start",):
                    self._planned_sequence.append(item.strip())
            print(f"[BackwardOnly] Planned sequence: {self._planned_sequence}")
        except Exception as e:
            print(f"[BackwardOnly] Plan failed: {e}")
            # fallback: Retrieve → Finish
            self._planned_sequence = ["Retrieve[unknown]", "Finish[unknown]"]

    def forward(self):
        """계획된 시퀀스를 순서대로 실행한다. LLM 재추론 없이 고정 시퀀스 수행."""
        if self._plan_index < len(self._planned_sequence):
            action_str = self._planned_sequence[self._plan_index]
            self._plan_index += 1
        else:
            # 계획 소진 → Finish로 강제 종료
            action_str = f"Finish[{self.answer or 'unknown'}]"

        # scratchpad에 기록 (is_halted 체크용)
        self.scratchpad += f"\nActionPath {self.step_n}: (backward-plan step)"
        self.scratchpad += f"\nThought {self.step_n}: Following backward plan."
        self.scratchpad += f"\nAction {self.step_n}: {action_str}"
        print(f"Action {self.step_n}: {action_str} [backward-plan]")

        action_type, argument = parse_action(action_str)
        self.action_history.append(action_type)
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        """분석용 프롬프트 (실행에는 사용되지 않음)."""
        return (
            f"[BackwardOnly] Question: {self.question}\n"
            f"Backward Plan:\n{self.backward_plan}\n"
            f"Executed sequence: {self._planned_sequence}\n"
            f"Scratchpad:{self.scratchpad}"
        )

    def get_first_action(self) -> str:
        return self.action_history[0] if self.action_history else ""

    def get_action_sequence(self) -> list[str]:
        return list(self.action_history)


def create_agent_pair(question: str, key: str, llm, backward_llm=None):
    """
    동일한 질문에 대해 baseline과 bidirectional agent를 함께 생성한다.
    실험에서 paired comparison에 사용.
    """
    baseline = BaselineKnowAgent(question=question, key=key, llm=llm)
    bidirectional = BidirectionalKnowAgent(
        question=question,
        key=key,
        llm=llm,
        backward_llm=backward_llm,
    )
    return baseline, bidirectional


def create_ablation_trio(question: str, key: str, llm, backward_llm=None):
    """
    3-way ablation을 위한 (A, B, C) agent 트리오를 생성한다.
    A: Forward Only, B: Backward Only, C: Bidirectional
    """
    agent_a = BaselineKnowAgent(question=question, key=key, llm=llm)
    agent_b = BackwardOnlyAgent(
        question=question, key=key, llm=llm, backward_llm=backward_llm
    )
    agent_c = BidirectionalKnowAgent(
        question=question, key=key, llm=llm, backward_llm=backward_llm
    )
    return agent_a, agent_b, agent_c
