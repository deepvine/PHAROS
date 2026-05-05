"""
BackwardPlanningAgent

주어진 질문에 대해 역방향 계획(Backward Plan)을 생성하는 모듈.
- 목표: Finish[answer]에서 Start로 역추론
- 출력: 역방향 action 시퀀스 + 각 단계의 reasoning
- 이 결과는 BidirectionalKnowAgent의 보조 입력으로 사용됨

Direction Bias 연구 맥락:
  정방향 agent는 Start에서 greedy하게 action을 선택하므로
  첫 번째 action 선택에 편향이 생긴다.
  역방향 계획은 goal-conditioned context를 제공하여 이를 완화한다.
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Path_Generation'))

from experiments.backward_prompts import (
    backward_planning_prompt,
    goal_directed_planning_prompt,
    minimal_planning_prompt,
    action_lift_planning_prompt,
    BACKWARD_FEWSHOT_EXAMPLES,
)


class BackwardPlanningAgent:
    """
    역방향 계획을 생성하는 에이전트.
    LLM을 사용해 질문으로부터 Finish→Start 방향의 action 계획을 만든다.
    """

    def __init__(self, llm, max_tokens: int = 512, mode: str = "rigid"):
        """
        Args:
            llm: LLM callable (langchain_openai_chatllm 등)
            max_tokens: 역방향 계획 생성 시 최대 토큰 수
            mode: "rigid" (Lv.1, Reversed sequence with arguments)
                 | "goal_directed" (사용자 의도 v2: backward reasoning + forward plan, args 없음)
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.mode = mode
        self.last_backward_plan = ""
        self.last_reversed_sequence = []
        self.last_first_action_type = ""  # goal_directed 모드용

    def generate_backward_plan(self, question: str) -> str:
        """
        질문에 대한 역방향 계획 텍스트를 생성한다.

        Returns:
            str: 역방향 계획 전체 텍스트
        """
        prompt_map = {
            "goal_directed": goal_directed_planning_prompt,
            "minimal":       minimal_planning_prompt,
            "action_lift":   action_lift_planning_prompt,
        }
        if self.mode in prompt_map:
            prompt = prompt_map[self.mode].format(question=question)
            tokens = 200 if self.mode == "minimal" else self.max_tokens
        else:
            prompt = backward_planning_prompt.format(question=question)
            tokens = self.max_tokens
        try:
            # stop=[]는 LLMWrapper에 "no stop, multi-line 응답 받기" 신호.
            plan = self.llm(prompt, stop=[], max_tokens=tokens)
        except TypeError:
            # Some LLM wrappers don't accept kwargs
            plan = self.llm(prompt)

        if isinstance(plan, dict):
            plan = plan.get("text", str(plan))

        self.last_backward_plan = plan.strip()
        # 모든 비-rigid 모드는 first action TYPE만 추출 (인자 없음).
        if self.mode in ("goal_directed", "minimal", "action_lift"):
            self.last_first_action_type = self._parse_first_action_type(plan)
            self.last_reversed_sequence = []
        else:
            self.last_reversed_sequence = self._parse_reversed_sequence(plan)
        return self.last_backward_plan

    def _parse_first_action_type(self, plan_text: str) -> str:
        """'First action: <Retrieve|Search|Either>' 또는 legacy 'First action type:' 라인 파싱."""
        # 새 minimal v3는 'First action:', goal_directed는 'First action type:'
        for pat in (r"First\s+action(?:\s+type)?[:\s]+(\w+)",):
            m = re.search(pat, plan_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                for valid in ("Retrieve", "Search", "Lookup", "Finish", "Either"):
                    if cand.lower() == valid.lower():
                        return valid
        return ""

    def _parse_reversed_sequence(self, plan_text: str) -> list[str]:
        """
        역방향 계획 텍스트에서 'Reversed sequence:' 라인을 파싱해
        action 목록을 반환한다.

        예시 입력:
          Reversed sequence: Start ← Retrieve[Milhouse] ← Lookup[named after] ← Finish[answer]
        반환:
          ['Retrieve[Milhouse]', 'Lookup[named after]', 'Finish[answer]']
        """
        sequence_match = re.search(
            r"Reversed sequence[:\s]+(.*?)(?:\n|$)", plan_text, re.IGNORECASE
        )
        if not sequence_match:
            return []

        seq_line = sequence_match.group(1).strip()
        # '←' 또는 '->' 또는 '=>'로 분리
        parts = re.split(r'[←→\-=>]+', seq_line)
        actions = []
        for part in parts:
            part = part.strip()
            # Start/Finish[answer] 같은 placeholder는 제외하고 실제 action만 수집
            if part and part.lower() != "start":
                actions.append(part)
        return actions

    def get_key_insight(self) -> str:
        """마지막으로 생성된 계획에서 Key insight를 추출한다."""
        if not self.last_backward_plan:
            return ""
        match = re.search(
            r"Key insight[:\s]+(.*?)(?:\n|$)", self.last_backward_plan, re.IGNORECASE
        )
        return match.group(1).strip() if match else ""

    def get_goal_type(self) -> str:
        """마지막으로 생성된 계획에서 GOAL 타입을 추출한다."""
        if not self.last_backward_plan:
            return "unknown"
        match = re.search(
            r"GOAL[:\s]+Finish\[(.+?)\]", self.last_backward_plan, re.IGNORECASE
        )
        return match.group(1).strip() if match else "unknown"

    def get_suggested_first_action(self) -> str:
        """
        역방향 계획에서 권장하는 첫 번째 정방향 action을 반환한다.
        - rigid 모드: reversed sequence의 첫 원소 (예: "Retrieve[Milhouse]")
        - goal_directed/minimal/action_lift: action type만 (예: "Retrieve")
        """
        if self.mode in ("goal_directed", "minimal", "action_lift"):
            return self.last_first_action_type
        if not self.last_reversed_sequence:
            return ""
        return self.last_reversed_sequence[0]


def create_backward_agent(llm) -> BackwardPlanningAgent:
    """팩토리 함수."""
    return BackwardPlanningAgent(llm)
