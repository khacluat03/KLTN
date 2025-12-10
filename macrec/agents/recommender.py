from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.base import ToolAgent
from macrec.tools import TextSummarizer
from macrec.utils import read_json, get_rm, parse_action

class Recommender(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 6)
        self.recommender = self.get_LLM(config=config)
        self.json_mode = self.recommender.json_mode
        self.latest_summary: str = ''
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'summarizer': TextSummarizer,
        }

    @property
    def summarizer(self) -> TextSummarizer:
        return self.tools['summarizer']

    @property
    def recommender_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['recommender_prompt_json']
        else:
            return self.prompts['recommender_prompt']

    @property
    def recommender_examples(self) -> str:
        if self.json_mode:
            return self.prompts['recommender_examples_json']
        else:
            return self.prompts['recommender_examples']

    def _build_recommender_prompt(self, **kwargs) -> str:
        return self.recommender_prompt.format(
            examples=self.recommender_examples,
            history=self.history,
            **kwargs
        )

    def _prompt_recommender(self, **kwargs) -> str:
        recommender_prompt = self._build_recommender_prompt(**kwargs)
        command = self.recommender(recommender_prompt)
        return command

    def reset(self) -> None:
        super().reset()
        self.latest_summary = ''

    def command(self, command: str, input: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'summarize':
            if self.latest_summary:
                observation = 'Summary already generated. Please call Finish[...] to output it.'
            else:
                observation = self.summarizer.summarize(text=input)
                self.latest_summary = observation
            log_head = ':violet[Summarize analysis...]\n- '
        elif action_type.lower() == 'finish':
            final_result = argument.strip() if isinstance(argument, str) else ''
            if self.latest_summary:
                final_result = self.latest_summary
            if not final_result:
                final_result = 'Recommender has no response to return.'
            if self.finished:
                observation = self.results
                log_head = ':violet[Finish already returned]:\n- '
            else:
                observation = self.finish(results=final_result)
                log_head = ':violet[Finish with recommendation]:\n- '
        else:
            observation = f'Unknown command type: {action_type}.'
        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def forward(self, preferences: str, analysis: str, *args, **kwargs) -> str:
        combined_input = f"Preferences: {preferences}\nAnalysis: {analysis}"
        while not self.is_finished():
            command = self._prompt_recommender(
                preferences=preferences,
                analysis=analysis,
                input=combined_input
            )
            self.command(command, input=combined_input)
        if not self.finished:
            return 'Recommender did not return any result.'
        return self.results

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if not isinstance(argument, dict) or 'preferences' not in argument or 'analysis' not in argument:
            return 'Invalid argument type. Must be a dict with "preferences" and "analysis" keys.'
        return self(preferences=argument['preferences'], analysis=argument['analysis'])


















