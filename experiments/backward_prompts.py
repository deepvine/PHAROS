"""
Backward Planning Prompts for BidirectionalKnowAgent

역방향 궤적 생성을 위한 프롬프트 모음.
목표 상태(Finish)에서 시작 상태(Start)로 역추론하여
정방향 action 생성의 direction bias를 측정/감소시키는 데 활용.
"""

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    class PromptTemplate:
        """Minimal PromptTemplate shim (no langchain dependency)."""
        def __init__(self, input_variables, template):
            self.input_variables = input_variables
            self.template = template

        def format(self, **kwargs):
            result = self.template
            for k, v in kwargs.items():
                result = result.replace("{" + k + "}", str(v))
            return result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 파일 로더 — 텍스트 파일에서 prompt 템플릿을 읽어 PromptTemplate으로 래핑.
# 외부 사용자가 prompt를 코드 수정 없이 편집할 수 있게 한다.
# ─────────────────────────────────────────────────────────────────────────────
import os as _os

_PROMPTS_DIR = _os.path.join(_os.path.dirname(__file__), "prompts")

def _load_prompt_text(filename: str) -> str:
    """experiments/prompts/<filename>에서 prompt 본문을 읽는다."""
    path = _os.path.join(_PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _load_prompt(filename: str, input_variables: list) -> "PromptTemplate":
    return PromptTemplate(
        input_variables=input_variables,
        template=_load_prompt_text(filename),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. Backward Planning Instruction
#    LLM에게 "질문에 답하기 위해 필요한 정보를 역방향으로 분해"하도록 요청.
#    이 역방향 계획은 forward agent의 보조 컨텍스트로 사용된다.
# ─────────────────────────────────────────────────────────────────────────────

BACKWARD_PLANNING_INSTRUCTION = """Your task is to create a BACKWARD PLAN for answering a question.
Instead of thinking forward (Start → Finish), reason BACKWARD from the goal:
  Finish[answer] ← What do I need to know? ← What actions provide that? ← Start

The action space is:
  Finish[answer]    - Concludes the task with the final answer
  Search[topic]     - Finds information via Bing Search
  Retrieve[entity]  - Retrieves a Wikipedia entity
  Lookup[keyword]   - Looks up a keyword in the last retrieved/searched passage

Instructions:
1. Identify the TYPE of answer needed (person, place, date, yes/no, etc.)
2. Decompose the question into required sub-facts (work backwards)
3. For each sub-fact, identify the action that would retrieve it
4. Produce a reversed action sequence: Finish ← ActionN ← ... ← Action1 ← Start

Here are examples of backward plans:

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Backward Plan:
  GOAL: Finish[person who Milhouse is named after]
  To answer with Finish, I need: the real person Milhouse Van Houten was named after
  To find that fact: Lookup[named after] in a passage about Milhouse
  To enable Lookup: Retrieve[Milhouse] to get his Wikipedia page
  Reversed sequence: Start ← Retrieve[Milhouse] ← Lookup[named after] ← Finish[answer]
  Key insight: This is a single-entity lookup; Retrieve then Lookup is most direct.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Backward Plan:
  GOAL: Finish[yes/no]
  To answer with Finish, I need: the professional fields of both people
  To find Leonid Levin's field: Search[Leonid Levin]
  To find Pavel Urysohn's field: Search[Pavel Urysohn]
  Reversed sequence: Start ← Search[Pavel Urysohn] ← Search[Leonid Levin] ← Finish[yes/no]
  Key insight: Both people need independent searches before comparison is possible.

(END OF EXAMPLES)

Now create a backward plan for the following question.
Provide the plan in exactly the same format as above.

Question: {question}
Backward Plan:"""

backward_planning_prompt = PromptTemplate(
    input_variables=["question"],
    template=BACKWARD_PLANNING_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bidirectional Forward Instruction
#    기존 KnowAgent instruction에 역방향 계획을 보조 힌트로 추가.
#    이 프롬프트가 direction bias 감소의 핵심 인터벤션이다.
# ─────────────────────────────────────────────────────────────────────────────

BIDIRECTIONAL_INSTRUCTION = """Your task is to answer a question using a specific graph-based method. You must navigate from the "Start" node to the "Finish" node by following the paths outlined in the graph. The correct path is a series of actions that will lead you to the answer.
The decision graph is constructed upon a set of principles known as "Action Knowledge", outlined as follows:
   Start:(Search, Retrieve)
   Retrieve:(Retrieve, Search, Lookup, Finish)
   Search:(Search, Retrieve, Lookup, Finish)
   Lookup:(Lookup, Search, Retrieve, Finish)
   Finish:()
Here's how to interpret the graph's Action Knowledge:
From "Start", you can initiate with either a "Search" or a "Retrieve" action.
At the "Retrieve" node, you have the options to persist with "Retrieve", shift to "Search", experiment with "Lookup", or advance to "Finish".
At the "Search" node, you can repeat "Search", switch to "Retrieve" or "Lookup", or proceed to "Finish".
At the "Lookup" node, you have the choice to keep using "Lookup", switch to "Search" or "Retrieve", or complete the task by going to "Finish".
The "Finish" node is the final action where you provide the answer and the task is completed.
Each node action is defined as follows:
(1) Retrieve[entity]: Retrieve the exact entity on Wikipedia and return the first paragraph if it exists. If not, return some similar entities for searching.
(2) Search[topic]: Use Bing Search to find relevant information on a specified topic, question, or term.
(3) Lookup[keyword]: Return the next sentence that contains the keyword in the last passage successfully found by Search or Retrieve.
(4) Finish[answer]: Return the answer and conclude the task.
As you solve the question using the above graph structure, interleave ActionPath, Thought, Action, and Observation steps. ActionPath documents the sequence of nodes you have traversed within the graph. Thought analyzes the current node to reveal potential next steps and reasons for the current situation.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

[BACKWARD PLANNING HINT]
Before executing, a goal-directed backward analysis was performed for this question.
Use this hint to guide your action choices and avoid unnecessary detours:
{backward_plan}
[END BACKWARD PLANNING HINT]

Question: {question}{scratchpad}"""

bidirectional_prompt = PromptTemplate(
    input_variables=["examples", "backward_plan", "question", "scratchpad"],
    template=BIDIRECTIONAL_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Goal-Directed Reasoning + Forward Plan (사용자 의도 v2)
#      "추론은 거꾸로(목표→수단), 출력은 forward(첫 단계부터)"
#      구체 인자(entity 이름·키워드)는 출력하지 않음 → anchoring 회피.
# ─────────────────────────────────────────────────────────────────────────────

GOAL_DIRECTED_PLANNING_INSTRUCTION = """Your task is to perform GOAL-DIRECTED REASONING for a question, then synthesize a FORWARD plan that an agent will execute.

The agent has these actions:
  Retrieve[entity]  - Wikipedia direct lookup (named entity preferred)
  Search[topic]     - Web search (broader, useful for fuzzy/open description)
  Lookup[keyword]   - Find keyword in last retrieved/searched passage
  Finish[answer]    - Conclude with the final answer

Two-phase process:
[Phase 1: Backward reasoning (cognitive only)]
  - Identify the FORM of the final answer (year, person name, yes/no, etc.)
  - Trace BACKWARD: what must hold to produce that answer? What information does that require? What action TYPE yields such information?
  - Continue until reaching Start.

[Phase 2: Forward plan synthesis]
  - Express the result as forward execution guidance.
  - DO NOT specify exact entity names or keywords as arguments — leave those to observation.
  - Output a recommended FIRST ACTION TYPE only (one of: Retrieve / Search).

Output format (exactly):
  Goal form: <type/shape of final answer>
  Backward reasoning: <2-4 short lines, no specific arguments>
  Forward plan:
    First action type: <Retrieve | Search>
    First action reason: <why this type, 1 sentence>
    Strategy: <high-level approach in 1-2 sentences>
    Estimated steps: <1-2 | 3-4 | 5+>
    Caveat: <optional, if any specific risk>

STRICT RULE: You may use action TYPES (Retrieve, Search, Lookup, Finish) in the reasoning, but NEVER include arguments like Retrieve[X] or Lookup[Y]. Arguments come from observations during execution.

Examples:

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Goal-Directed Plan:
  Goal form: A real person's name
  Backward reasoning:
    To Finish with a person name, I need a passage stating who Milhouse was named after.
    To obtain such a passage, I should perform Retrieve on the named entity (the character) — direct Wikipedia access is most efficient.
    Once retrieved, a Lookup may surface the specific "named after" fact within that passage.
  Forward plan:
    First action type: Retrieve
    First action reason: Milhouse is a named entity with a likely Wikipedia page.
    Strategy: Retrieve the entity, then Lookup the relevant phrase if needed.
    Estimated steps: 2-3
    Caveat: If the summary lacks the answer, follow up with Lookup or related-entity Retrieve.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Goal-Directed Plan:
  Goal form: Yes/no
  Backward reasoning:
    To Finish yes/no, I need profession info for both individuals.
    Each requires an independent information-gathering step.
    Both are named individuals; Retrieve is most direct (Search as fallback).
  Forward plan:
    First action type: Retrieve
    First action reason: Both are named entities likely on Wikipedia.
    Strategy: Gather profession of each entity in turn, then compare to decide yes/no.
    Estimated steps: 3-4
    Caveat: If a Retrieve fails to yield profession info, fall back to Search.

Question: Who was once considered the best kick boxer in the world but involved in controversies?
Goal-Directed Plan:
  Goal form: A person's name
  Backward reasoning:
    To Finish a person name, I need a passage identifying that specific kickboxer.
    No named entity is given; the description is fuzzy.
    A broad Search is more appropriate than direct Retrieve.
    Subsequent Lookup or follow-up Retrieve may narrow down the candidate.
  Forward plan:
    First action type: Search
    First action reason: No specific entity name in the question; broad search needed.
    Strategy: Search for "best kickboxer controversies" or similar; identify candidate; verify via Retrieve or Lookup.
    Estimated steps: 3-5
    Caveat: Multiple candidates plausible (Andrew Tate, Badr Hari, etc.); cross-validate before Finish.

(END OF EXAMPLES)

Now produce the goal-directed reasoning and forward plan for this question.

Question: {question}
Goal-Directed Plan:"""

goal_directed_planning_prompt = PromptTemplate(
    input_variables=["question"],
    template=GOAL_DIRECTED_PLANNING_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Minimal Strategic Prior (Option A)
#      Sequencing/multi-step plan 완전 제거. 첫 action type + 짧은 reason만.
# ─────────────────────────────────────────────────────────────────────────────

MINIMAL_PLANNING_INSTRUCTION = """You are a strategic prior for an agent that has TWO complementary information actions:

  - Retrieve[entity]:  fetch a Wikipedia article by exact entity name
                      (best when the question names a specific entity that
                       is likely to have its own Wikipedia page).
  - Search[query]:    perform a web search via Tavily for raw snippets
                      (best when the question describes something fuzzy,
                       commonsense, comparative without clear entities,
                       or recent/dynamic information).

Output exactly 4 lines:
  Question type: <one short label>
  Expected answer form: <specific form of the final answer>
  First action: <Retrieve | Search | Either>
  Reason: <one-sentence justification>

The "Expected answer form" should be as concrete as possible — e.g.,
  "a year (4-digit number)", "a person's full name",
  "yes or no", "a film title", "a city name", "a profession (one word)".
This helps the agent format Finish[answer] correctly.

The "First action" rubric:
  - Retrieve: question names one or more specific entities AND the answer
              is a fact recorded in those entities' Wikipedia articles.
  - Search:   question is descriptive without named entities, OR requires
              commonsense / yes-no strategic reasoning, OR involves recent/
              niche topics, OR broad context speeds up comparison.
  - Either:   both seem reasonable (e.g., named entity exists but answer
              also benefits from broader context).

DO NOT output multi-step plans, sequences, conditional logic, specific
entity names, estimated step counts, or caveats.

Examples:

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Strategic Prior:
  Question type: single-entity factual
  Expected answer form: a real person's full name
  First action: Retrieve
  Reason: Question names a specific character (Milhouse) whose Wikipedia article should contain the named-after fact.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Strategic Prior:
  Question type: comparison (two named entities)
  Expected answer form: yes or no
  First action: Either
  Reason: Both are named entities (Retrieve viable), but a Search comparing both can return the answer in one step.

Question: Could a llama birth twice during the year 2018?
Strategic Prior:
  Question type: yes/no commonsense (gestation period reasoning)
  Expected answer form: yes or no
  First action: Search
  Reason: Answer requires combining gestation-period knowledge with calendar arithmetic; no single Wikipedia entity covers it.

Question: Did Aristotle use a laptop?
Strategic Prior:
  Question type: yes/no commonsense (chronology check)
  Expected answer form: yes or no
  First action: Search
  Reason: Anachronism check; broad context is more efficient than targeted Retrieve.

Question: Who was once considered the best kickboxer with controversies?
Strategic Prior:
  Question type: open-description (no named entity)
  Expected answer form: a person's full name
  First action: Search
  Reason: Question lacks a specific entity; broad search needed to identify the candidate person.

Question: Which film won the Academy Award for Best Picture in 1994?
Strategic Prior:
  Question type: single-record factual (year-specific)
  Expected answer form: a film title
  First action: Either
  Reason: A targeted Retrieve on the relevant Awards article works; a Search on the same query also returns the answer directly.

Question: In which stadium do the teams owned by Myra Kraft's husband play?
Strategic Prior:
  Question type: chained factual (multi-hop)
  Expected answer form: a stadium name (proper noun)
  First action: Search
  Reason: Single Search query can surface both husband identity and stadium in one step; targeted Retrieve requires multiple hops.

(END OF EXAMPLES)

Question: {question}
Strategic Prior:"""

# 기본 minimal prompt는 텍스트 파일에서 로드한다 (experiments/prompts/janus_v3_minimal.txt).
# inline MINIMAL_PLANNING_INSTRUCTION 변수는 fallback으로 보존.
try:
    minimal_planning_prompt = _load_prompt(
        "janus_v3_minimal.txt", input_variables=["question"]
    )
except (FileNotFoundError, OSError):
    minimal_planning_prompt = PromptTemplate(
        input_variables=["question"],
        template=MINIMAL_PLANNING_INSTRUCTION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.7  Action-Lift Backward Chaining (Lv.2)
#      Goal에서 거꾸로 추론하되 구체 인자 없이 action TYPE 시퀀스만 도출.
#      "lifted backward chaining" — STRIPS-style 추론 보존, anchoring 회피.
# ─────────────────────────────────────────────────────────────────────────────

ACTION_LIFT_PLANNING_INSTRUCTION = """Your task is BACKWARD CHAINING at the action TYPE level.

Reason from the goal backward, identifying what action TYPES are needed.
DO NOT include specific entity names or keyword arguments.
Use action types only: Retrieve, Search, Lookup, Finish.

Output format:
  Goal type: <answer form, e.g. "person name" / "year" / "yes-no">
  Backward chain (types only):
    Finish[<type>] requires: <action type that produces required info>
    That requires: <prior action type, if any>
    Initial: <Retrieve | Search>
  First action: <Retrieve | Search>

Examples:

Question: Who was Milhouse named after?
Backward Chain:
  Goal type: person name
  Backward chain (types only):
    Finish[person name] requires: a passage stating the named-after fact
    That requires: Lookup within an entity's Wikipedia page
    That requires: Retrieve the named entity
    Initial: Retrieve
  First action: Retrieve

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Backward Chain:
  Goal type: yes/no
  Backward chain (types only):
    Finish[yes/no] requires: profession info on each named individual
    That requires: independent Retrieve on each entity
    Initial: Retrieve
  First action: Retrieve

Question: Best kickboxer involved in controversies?
Backward Chain:
  Goal type: person name
  Backward chain (types only):
    Finish[person name] requires: a passage identifying a kickboxer with controversies
    That requires: a candidate-discovery phase (no named entity)
    Initial: Search
  First action: Search

(END OF EXAMPLES)

Question: {question}
Backward Chain:"""

action_lift_planning_prompt = PromptTemplate(
    input_variables=["question"],
    template=ACTION_LIFT_PLANNING_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.8  Freedom-clause Forward Prompt
#      Backward plan은 goal_directed 그대로 쓰되, forward 에이전트에 명시적으로
#      "Plan은 advisory; observation에 답 있으면 Finish 즉시" 지시 추가.
# ─────────────────────────────────────────────────────────────────────────────

BIDIRECTIONAL_INSTRUCTION_WITH_FREEDOM = """Your task is to answer a question using a specific graph-based method. You must navigate from the "Start" node to the "Finish" node by following the paths outlined in the graph. The correct path is a series of actions that will lead you to the answer.
The decision graph is constructed upon a set of principles known as "Action Knowledge", outlined as follows:
   Start:(Search, Retrieve)
   Retrieve:(Retrieve, Search, Lookup, Finish)
   Search:(Search, Retrieve, Lookup, Finish)
   Lookup:(Lookup, Search, Retrieve, Finish)
   Finish:()
Here's how to interpret the graph's Action Knowledge:
From "Start", you can initiate with either a "Search" or a "Retrieve" action.
At the "Retrieve" node, you have the options to persist with "Retrieve", shift to "Search", experiment with "Lookup", or advance to "Finish".
At the "Search" node, you can repeat "Search", switch to "Retrieve" or "Lookup", or proceed to "Finish".
At the "Lookup" node, you have the choice to keep using "Lookup", switch to "Search" or "Retrieve", or complete the task by going to "Finish".
The "Finish" node is the final action where you provide the answer and the task is completed.
Each node action is defined as follows:
(1) Retrieve[entity]: Retrieve the exact entity on Wikipedia and return the first paragraph if it exists. If not, return some similar entities for searching.
(2) Search[topic]: Use Bing Search to find relevant information on a specified topic, question, or term.
(3) Lookup[keyword]: Return the next sentence that contains the keyword in the last passage successfully found by Search or Retrieve.
(4) Finish[answer]: Return the answer and conclude the task.
As you solve the question using the above graph structure, interleave ActionPath, Thought, Action, and Observation steps. ActionPath documents the sequence of nodes you have traversed within the graph. Thought analyzes the current node to reveal potential next steps and reasons for the current situation.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

[BACKWARD PLANNING HINT]
Before executing, a goal-directed backward analysis was performed for this question.
Use this hint as a soft prior to bias your first action choice.
{backward_plan}
[END BACKWARD PLANNING HINT]

[IMPORTANT: PLAN IS ADVISORY]
- The plan above is a PRIOR, not a script. It does not bind your subsequent steps.
- If, at any step, the current observation already contains enough information to answer
  the question, IMMEDIATELY use Finish[answer] — do NOT execute additional plan steps.
- Trust observations over the plan when they conflict.

Question: {question}{scratchpad}"""

bidirectional_prompt_with_freedom = PromptTemplate(
    input_variables=["examples", "backward_plan", "question", "scratchpad"],
    template=BIDIRECTIONAL_INSTRUCTION_WITH_FREEDOM,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Few-shot examples for backward planning (extended)
# ─────────────────────────────────────────────────────────────────────────────

BACKWARD_FEWSHOT_EXAMPLES = """Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Backward Plan:
  GOAL: Finish[person who Milhouse is named after]
  To answer with Finish, I need: the real person Milhouse Van Houten was named after
  To find that fact: Lookup[named after] in a passage about Milhouse
  To enable Lookup: Retrieve[Milhouse] to get his Wikipedia page
  Reversed sequence: Start ← Retrieve[Milhouse] ← Lookup[named after] ← Finish[answer]
  Key insight: This is a single-entity lookup; Retrieve then Lookup is most direct.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Backward Plan:
  GOAL: Finish[yes/no]
  To answer with Finish, I need: the professional fields of both people
  To find Leonid Levin's field: Search[Leonid Levin]
  To find Pavel Urysohn's field: Search[Pavel Urysohn]
  Reversed sequence: Start ← Search[Pavel Urysohn] ← Search[Leonid Levin] ← Finish[yes/no]
  Key insight: Both people need independent searches before comparison is possible."""
