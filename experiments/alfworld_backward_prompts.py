"""
ALFWorld Backward Planning Prompts

ALFWorld의 6가지 태스크 유형에 대한 역방향 계획 프롬프트.
HotpotQA와 달리 물리적 환경에서의 action 시퀀스를 역추론한다.

태스크 유형:
  - put:     물체를 집어 특정 위치에 놓기
  - clean:   물체를 세척 후 배치
  - heat:    물체를 가열 후 배치
  - cool:    물체를 냉각 후 배치
  - examine: 특정 조명 아래서 물체 확인
  - puttwo:  두 물체를 같은 위치에 배치

Direction Bias 맥락:
  ALFWorld에서 정방향 에이전트는 태스크 목표와 무관하게
  "가장 가까운 위치로 이동"하는 편향이 나타날 수 있다.
  역방향 계획은 목표 위치에서 역추론하여 더 효율적인 경로를 제시한다.
"""

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    class PromptTemplate:
        def __init__(self, input_variables, template):
            self.input_variables = input_variables
            self.template = template
        def format(self, **kwargs):
            result = self.template
            for k, v in kwargs.items():
                result = result.replace("{" + k + "}", str(v))
            return result


# ─────────────────────────────────────────────────────────────────────────────
# Backward Planning Instruction for ALFWorld
# ─────────────────────────────────────────────────────────────────────────────

ALFWORLD_BACKWARD_INSTRUCTION = """Your task is to create a BACKWARD PLAN for completing a household task.
Instead of thinking forward (find object → navigate → place object), reason BACKWARD from the goal state:
  Goal achieved ← What must be done last? ← What must come before? ← Initial state

The action space is:
  go to [receptacle]          - Move to a location
  take [object] from [receptacle] - Pick up an object
  put [object] in/on [receptacle] - Place an object
  open [receptacle]           - Open a container
  clean [object] with [receptacle] - Clean using sink/basin
  heat [object] with [receptacle]  - Heat using microwave/stove
  cool [object] with [receptacle]  - Cool using fridge
  examine [object] with [light]    - Examine under a light source
  use [light source]               - Use a light for examination

Instructions:
1. Identify the FINAL STATE required (what does success look like?)
2. Work backwards: what action produces that final state?
3. What must happen before that action?
4. Continue until you reach the initial state
5. Produce a reversed action sequence: Goal ← ActionN ← ... ← Action1 ← Start

Here are examples of backward plans for ALFWorld tasks:

Task: put some peppershaker on diningtable.
Backward Plan:
  GOAL: peppershaker is on diningtable
  Last action needed: put peppershaker in/on diningtable
  Before that: go to diningtable (to be at the right location)
  Before that: take peppershaker from [its location]
  Before that: go to [location of peppershaker]
  Reversed sequence: Start ← go to [peppershaker location] ← take peppershaker ← go to diningtable ← put peppershaker ← Goal
  Key insight: Find peppershaker first, then navigate to diningtable. Check likely locations (countertop, drawer) first.

Task: clean some mug and put it in coffeemachine.
Backward Plan:
  GOAL: clean mug is in coffeemachine
  Last action needed: put mug in/on coffeemachine
  Before that: go to coffeemachine
  Before that: clean mug with sinkbasin
  Before that: go to sinkbasin (holding mug)
  Before that: take mug from [its location]
  Before that: go to [location of mug]
  Reversed sequence: Start ← go to [mug location] ← take mug ← go to sinkbasin ← clean mug ← go to coffeemachine ← put mug ← Goal
  Key insight: Must clean before placing. Mug likely in cabinet or countertop.

(END OF EXAMPLES)

Now create a backward plan for the following task.
Provide the plan in exactly the same format as above.

Task: {task_description}
Backward Plan:"""

alfworld_backward_prompt = PromptTemplate(
    input_variables=["task_description"],
    template=ALFWORLD_BACKWARD_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Task-type별 few-shot backward plan 예시
# ─────────────────────────────────────────────────────────────────────────────

ALFWORLD_BACKWARD_EXAMPLES = {
    "put": """Task: put some peppershaker on diningtable.
Backward Plan:
  GOAL: peppershaker is on diningtable
  Last action needed: put peppershaker in/on diningtable
  Before that: go to diningtable
  Before that: take peppershaker from its current location
  Before that: go to peppershaker's location (check countertop, drawer)
  Reversed sequence: Start ← go to [src] ← take peppershaker ← go to diningtable ← put ← Goal
  Key insight: Locate object first. Diningtable is the destination.""",

    "clean": """Task: clean some mug and put it in coffeemachine.
Backward Plan:
  GOAL: clean mug is in coffeemachine
  Last action needed: put mug in/on coffeemachine
  Before that: go to coffeemachine
  Before that: clean mug with sinkbasin
  Before that: go to sinkbasin (already holding mug)
  Before that: take mug from location, go to its location
  Reversed sequence: Start ← go to [mug] ← take mug ← go to sink ← clean ← go to coffeemachine ← put ← Goal
  Key insight: clean step is mandatory before placement.""",

    "heat": """Task: heat some egg and put it in fridge.
Backward Plan:
  GOAL: heated egg is in fridge
  Last action needed: put egg in/on fridge
  Before that: go to fridge
  Before that: heat egg with microwave
  Before that: go to microwave (holding egg)
  Before that: take egg from its location
  Reversed sequence: Start ← go to [egg] ← take egg ← go to microwave ← heat ← go to fridge ← put ← Goal
  Key insight: heat before place. Microwave is the heating receptacle.""",

    "cool": """Task: cool some lettuce and put it in countertop.
Backward Plan:
  GOAL: cooled lettuce is on countertop
  Last action needed: put lettuce in/on countertop
  Before that: go to countertop
  Before that: cool lettuce with fridge
  Before that: go to fridge (holding lettuce)
  Before that: take lettuce from location
  Reversed sequence: Start ← go to [lettuce] ← take lettuce ← go to fridge ← cool ← go to countertop ← put ← Goal
  Key insight: cool step mandatory. Fridge is the cooling receptacle.""",

    "examine": """Task: examine the alarmclock with the desklamp.
Backward Plan:
  GOAL: alarmclock examined under desklamp
  Last action needed: use desklamp (while holding alarmclock near it)
  Before that: go to desklamp location
  Before that: take alarmclock from its location
  Before that: go to alarmclock location
  Reversed sequence: Start ← go to [alarmclock] ← take alarmclock ← go to desklamp ← use desklamp ← Goal
  Key insight: Need to be at light source location with object in hand.""",

    "puttwo": """Task: put two soapbottle in cabinet.
Backward Plan:
  GOAL: two soapbottles are in cabinet
  Last action needed: put 2nd soapbottle in/on cabinet
  Before that: go to cabinet (2nd time)
  Before that: take 2nd soapbottle from its location
  Before that: put 1st soapbottle in cabinet (first trip)
  Before that: take 1st soapbottle, go to cabinet
  Reversed sequence: Start ← find soapbottle1 ← put in cabinet ← find soapbottle2 ← put in cabinet ← Goal
  Key insight: Two separate trips needed. Check countertop, shelf for both objects.""",
}
