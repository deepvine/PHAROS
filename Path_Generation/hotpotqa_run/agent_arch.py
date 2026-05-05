"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import random
import re, string, os
import json 
import time
import tiktoken
from langchain.llms.base import BaseLLM
from langchain.docstore import Wikipedia
from langchain.llms import OpenAI
from langchain.docstore.base import Docstore
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import PromptTemplate
from collections import Counter
from openai import OpenAI as _OpenAIClient


from hotpotqa_run.pre_prompt import knowagent_prompt
from hotpotqa_run.fewshots import KNOWAGENT_EXAMPLE

from hotpotqa_run.llms import token_enc

# Search 클라이언트 (Bing v7 deprecation 후 Tavily로 전환).
# Tavily는 raw search snippets 반환 — Bing과 같은 format으로 LLM 후처리 없음.
from hotpotqa_run.config import TAVILY_API_KEY
try:
    from tavily import TavilyClient
    _TAVILY_CLIENT = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY != "empty_key" else None
except ImportError:
    _TAVILY_CLIENT = None

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$' 
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        action_type, argument = fuzzy_parse_action(string)
        return action_type, argument
        
def fuzzy_parse_action(text):
    text = text.strip(' ').strip('.')
    pattern = r'^(\w+)\[(.+)\]'
    match = re.match(pattern, text)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return text, ''

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = token_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(token_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
  
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


class BaseAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 llm: BaseLLM,
                 context_len: int = 2000,
                 max_steps: int= 10,
                 docstore: Docstore = Wikipedia()
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = ""
        self.examples = ""
        self.context_len = context_len
        self.run_error = False
        self.name = "Base_HotPotQA_run_Agent"
        self.pre_action = ''
        self.docstore = DocstoreExplorer(docstore) # Search, Lookup
        self.bingsearch_results = ''
        self.search_results_num = 3
        self.llm = llm
        
        self.enc = token_enc
        self.__reset_agent()
    
    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished() and not self.run_error:
            self.step()

    def prompt_agent(self) -> str:
        import os as _os
        _prompt = self._build_agent_prompt()
        if _os.environ.get("KNOWAGENT_DEBUG"):
            print(f"[DEBUG prompt tail] ...{_prompt[-120:]!r}")
        try:
            generation = self.llm(_prompt)
            self.check_run_error(generation)
        except Exception as _e:
            if _os.environ.get("KNOWAGENT_DEBUG"):
                print(f"[DEBUG prompt_agent EXCEPTION] {type(_e).__name__}: {_e}")
            generation = ""
        if _os.environ.get("KNOWAGENT_DEBUG"):
            print(f"[DEBUG raw LLM output] repr={generation!r}")
        return format_step(generation)
 
    def check_run_error(self, text):
        if text in ["No response"]:
            self.run_error = True
            
    def is_finished(self) -> bool:
        return self.finished
    
    def reward(self) -> float:
        return f1_score(self.answer, self.key)   
    
    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps)
                or (len(self.enc.encode(self._build_agent_prompt())) > self.context_len)
                ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''
        self._consecutive_empty = 0  # cascade detection counter

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def _strip_echoed_prefix(self, text: str, prefix: str) -> str:
        t = text.lstrip()
        if t.startswith(prefix):
            t = t[len(prefix):].lstrip(': ').lstrip()
        return t

    def _single_step_generate(self) -> tuple[str, str, str]:
        """Chat 모델용: ActionPath/Thought/Action을 한 번의 LLM 호출로 생성하고 파싱한다.

        반환: (action_path_text, thought_text, action_text)
        """
        self.scratchpad += f'\nActionPath {self.step_n}:'
        full = ""
        try:
            # bare "Observation:" 제거 — Thought의 "From Observation N:" 인용 패턴에 걸려
            # 생성이 즉시 중단되는 cascade halt 방지
            full = self.llm(
                self._build_agent_prompt(),
                stop=[f"\nObservation {self.step_n}:"],
                max_tokens=600,
            )
        except TypeError:
            try:
                full = self.llm(self._build_agent_prompt())
            except Exception:
                full = ""
        except Exception:
            full = ""

        # Parse 3개 필드 (줄 단위로 정규식)
        ap_match = re.search(
            rf'ActionPath\s*{self.step_n}\s*:\s*(.*?)(?=\n\s*Thought\s*{self.step_n}|\Z)',
            full, re.DOTALL)
        th_match = re.search(
            rf'Thought\s*{self.step_n}\s*:\s*(.*?)(?=\n\s*Action\s*{self.step_n}|\Z)',
            full, re.DOTALL)
        # 숫자 없는 "Action:" 형식도 허용 — 모델이 번호를 누락해도 파싱 성공
        ac_match = re.search(
            rf'Action\s*{self.step_n}\s*:\s*([^\n]+)',
            full) or re.search(
            r'Action\s*:\s*([^\n]+)',
            full)

        action_path = ap_match.group(1).strip() if ap_match else ""
        thought     = th_match.group(1).strip() if th_match else ""
        action_text = ac_match.group(1).strip() if ac_match else ""

        import os as _os
        if _os.environ.get("KNOWAGENT_DEBUG"):
            print(f"[DEBUG single_step] raw={full[:200]!r}")
            print(f"[DEBUG single_step] parsed path={action_path!r} thought={thought[:50]!r} action={action_text!r}")

        # scratchpad 재구성
        # ActionPath 뒤에 나머지 추가
        self.scratchpad += ' ' + action_path
        self.scratchpad += f'\nThought {self.step_n}: ' + thought
        self.scratchpad += f'\nAction {self.step_n}: ' + action_text
        print(f'ActionPath {self.step_n}: {action_path}')
        print(f'Thought {self.step_n}: {thought[:80]}')
        print(f'Action {self.step_n}: {action_text}')
        return action_path, thought, action_text

    def _actionpath(self):
        self.scratchpad += f'\nActionPath {self.step_n}:'
        action_path = self.prompt_agent()
        action_path = self._strip_echoed_prefix(action_path, f'ActionPath {self.step_n}')
        self.scratchpad += ' ' + action_path
        print(self.scratchpad.split('\n')[-1])

    def _think(self):
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.prompt_agent()
        thought = self._strip_echoed_prefix(thought, f'Thought {self.step_n}')
        self.scratchpad += ' ' + thought
        print(self.scratchpad.split('\n')[-1])

    def _action(self):
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        action = self._strip_echoed_prefix(action, f'Action {self.step_n}')
        pattern = re.compile(r'\s+(?=\[)')
        action = pattern.sub('', action)
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        return action_type, argument

    def _bingsearch(self, argument):
        """Bing Search API v7 deprecation 대응 — Tavily로 raw snippets 호출.
        Tavily는 LLM 후처리 없이 search 결과(title/url/content)를 그대로 반환,
        Bing snippets와 같은 성격을 유지한다.
        결과 포맷은 search_lookup과 호환되도록 [{"snippet": text, "title": url}]로 정규화."""
        if _TAVILY_CLIENT is None:
            return f'Search is unavailable in this environment. Use Retrieve[{argument}] instead to query Wikipedia.'
        result = ''
        try:
            resp = _TAVILY_CLIENT.search(
                argument,
                search_depth="basic",
                max_results=self.search_results_num,
            )
            results = resp.get("results", [])
            self.bingsearch_results = [
                {"snippet": r.get("content", ""), "title": r.get("title", "") or r.get("url", "")}
                for r in results
            ]
            if results:
                top = results[0]
                snippet = (top.get("content") or "")[:500]
                title = top.get("title") or top.get("url") or "tavily"
                result = f"{title}: {snippet}" if snippet else snippet or "No results returned."
            else:
                result = "No results returned."
        except Exception:
            self.scratchpad += 'Search error,please try again'
        return result

    def search_lookup(self, argument):
        if self.bingsearch_results == '':
            return "You need to search first."
        else:
            lookups = []
            try:
                for res in self.bingsearch_results:
                    argument_words = argument.lower().split()
                    snippet_lower = res["snippet"].lower()
                    name_lower = res["title"].lower()
                    if all(word in snippet_lower or word in name_lower for word in argument_words):
                        lookups.append(snippet_lower.replace("<b>", "").replace("</b>", ""))
                    if lookups == []:
                        return "No results found."
                    else:
                        return lookups[0]
            except:
                return "No results found."

    def step(self) -> None:

        # agent forward
        ret = self.forward()
        if ret:
            action_type, argument = ret[0], ret[1]
        else:
            action_type = ret

        # cascade detection: 빈 action이 2번 연속이면 즉시 종료
        # (stop sequence 오발동으로 생긴 empty step이 scratchpad에 쌓여
        #  모델이 빈 패턴을 모방하는 self-reinforcing halt 방지)
        if not action_type:
            self._consecutive_empty += 1
            if self._consecutive_empty >= 2:
                self.answer = 'information unavailable'
                self.finished = True
                self.scratchpad += f'\nObservation {self.step_n}: Cascade empty — forced stop.'
                self.step_n += 1
                return
        else:
            self._consecutive_empty = 0

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.pre_action = "Search"
                tmp = self._bingsearch(format_step(argument))
                self.scratchpad += format_step(tmp)
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not complete the Bing search, please try again.'

        elif action_type == 'Retrieve':
            try:
                self.pre_action = "Retrieve"
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            if self.pre_action == "Retrieve":
                try:
                    self.scratchpad += format_step(self.docstore.lookup(argument))
                except ValueError:
                    self.scratchpad += f'The last page Retrieved was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
            elif self.pre_action == "Search":
                try:
                    self.scratchpad += format_step(self.search_lookup(argument))
                except ValueError:
                    self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] Retrieve[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1
    
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    
class KnowAgentHotpotQA(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = KNOWAGENT_EXAMPLE
        self.agent_prompt = knowagent_prompt
        self.name = "KnowAgentHotpotQA"

    def forward(self):
        _action_path, _thought, action_text = self._single_step_generate()
        pattern = re.compile(r'\s+(?=\[)')
        action_text = pattern.sub('', action_text)
        action_type, argument = parse_action(action_text)
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
        
def get_agent(agent_name):
    if agent_name in ["KnowAgentHotpotQA"]:
        return KnowAgentHotpotQA
    else:
        return None