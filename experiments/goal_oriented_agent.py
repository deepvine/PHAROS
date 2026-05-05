"""
GoalOrientedKnowAgent

매 스텝의 Thought를 backward-chaining 구조 (Goal / Need / Have / Gap / -> next action)로
강제하는 agent. 별도 backward plan 생성 없이 단일 LLM 호출로 동작 (BaselineKnowAgent와 호출 수 동일).

vs BaselineKnowAgent:
  - 동일: forward only, 1 call/step, 동일 그래프 제약, 동일 action 정의
  - 차이: prompt template만 교체 (Thought 형식이 backward-chain 4줄로 고정)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Path_Generation"))

import re
from langchain.prompts import PromptTemplate

from hotpotqa_run.agent_arch import BaseAgent, parse_action


_PROMPT_PATH = Path(__file__).parent / "prompts" / "goal_oriented_v2_5_en.txt"


def _load_goal_oriented_template() -> PromptTemplate:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    # 텍스트 안에 예시들이 baked-in 되어 있어 input_variables는 question/scratchpad만.
    return PromptTemplate(
        input_variables=["question", "scratchpad"],
        template=text,
    )


class GoalOrientedKnowAgent(BaseAgent):
    """
    Backward-chaining Thought 구조를 prompt-level로 강제하는 forward agent.
    BaselineKnowAgent와 동일한 인터페이스 (action_history 기록 포함)이며,
    bias_metrics.py / run_experiment.py 와 호환된다.
    """

    def __init__(
        self,
        question: str,
        key: str,
        llm,
        context_len: int = 6000,
    ) -> None:
        super().__init__(question, key, llm, context_len)
        self.agent_prompt = _load_goal_oriented_template()
        self.examples = ""  # 사용 안 함 (prompt 안에 이미 예시 포함)
        self.name = "GoalOrientedKnowAgent"
        self.action_history: list[str] = []

    def forward(self):
        _ap, _th, action_text = self._single_step_generate()
        action_text = re.sub(r'\s+(?=\[)', '', action_text)
        action_type, argument = parse_action(action_text)
        self.action_history.append(action_type)
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def get_first_action(self) -> str:
        return self.action_history[0] if self.action_history else ""

    def get_action_sequence(self) -> list[str]:
        return list(self.action_history)
