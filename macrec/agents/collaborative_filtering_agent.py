"""
Collaborative Filtering Agent
Đặc biệt dành cho Collaborative Filtering recommendations
"""

from typing import Any, List, Optional
from loguru import logger
import json

from macrec.agents.base import ToolAgent
from macrec.tools.info_database import InfoDatabase
from macrec.tools.interaction import InteractionRetriever
from macrec.tools.collaborative_filtering import CollaborativeFiltering
from macrec.utils import read_json, get_rm, parse_action


class CollaborativeFilteringAgent(ToolAgent):
    """
    Agent chuyên về Collaborative Filtering recommendations
    Sử dụng user-item interaction patterns để recommend
    """

    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 8)
        self.default_top_k = get_rm(config, 'default_top_k', 10)
        # Remove agent-specific config before passing to LLM
        llm_config = config.copy()
        llm_config.pop('tool_config', None)
        llm_config.pop('max_turns', None)
        llm_config.pop('default_top_k', None)
        self.cf_agent = self.get_LLM(config=llm_config)
        self.json_mode = self.cf_agent.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
            'cf_tool': CollaborativeFiltering,
        }

    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']

    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']

    @property
    def cf_tool(self) -> CollaborativeFiltering:
        return self.tools['cf_tool']

    @property
    def cf_agent_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['cf_agent_prompt_json']
        else:
            return self.prompts['cf_agent_prompt']

    @property
    def cf_agent_examples(self) -> str:
        if self.json_mode:
            return self.prompts['cf_agent_examples_json']
        else:
            return self.prompts['cf_agent_examples']

    @property
    def cf_agent_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['cf_agent_fewshot_json']
        else:
            return self.prompts['cf_agent_fewshot']

    @property
    def hint(self) -> str:
        if 'cf_agent_hint' not in self.prompts:
            return ''
        return self.prompts['cf_agent_hint']

    def _build_cf_agent_prompt(self, **kwargs) -> str:
        return self.cf_agent_prompt.format(
            examples=self.cf_agent_examples,
            fewshot=self.cf_agent_fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )

    def _prompt_cf_agent(self, **kwargs) -> str:
        cf_agent_prompt = self._build_cf_agent_prompt(**kwargs)
        command = self.cf_agent(cf_agent_prompt)
        return command

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)

        if action_type.lower() == 'recommend' or action_type.lower() == 'cf':
            # CF Recommendation
            try:
                if self.json_mode:
                    user_id = argument.get('user_id', argument)
                    top_k = argument.get('top_k', self.default_top_k)
                    candidates = argument.get('candidates', None)
                else:
                    # Parse "user_id,top_k,candidates" format hoặc "user_id,top_k"
                    parts = argument.split(',')
                    user_id = int(parts[0].strip())
                    top_k = int(parts[1].strip()) if len(parts) > 1 else self.default_top_k
                    candidates = None
                    if len(parts) > 2:
                        # Parse candidates list from string like "[1,2,3]" hoặc "1,2,3"
                        candidates_str = ','.join(parts[2:]).strip()
                        if candidates_str.startswith('[') and candidates_str.endswith(']'):
                            # JSON list format
                            candidates = eval(candidates_str)
                        else:
                            # Comma separated format
                            candidates = [int(x.strip()) for x in candidates_str.split(',') if x.strip()]

                # Get CF recommendations
                recommendations = self.cf_tool.recommend_items_user_based(
                    user_id=user_id,
                    n_items=top_k,
                    candidates=candidates
                )

                # Get item details for better presentation
                if recommendations:
                    detailed_recs = []
                    for item_id in recommendations[:top_k]:
                        item_info = self.info_retriever.item_info(item_id=item_id)
                        detailed_recs.append(f"{item_id}: {item_info}")
                    observation = f"CF Recommendations for User {user_id} (top {top_k}):\n" + "\n".join(detailed_recs)
                else:
                    observation = f"No CF recommendations available for user {user_id}"

                log_head = f':violet[CF Recommendation for User] :red[{user_id}]:violet[...]\n- '

            except (ValueError, KeyError, IndexError) as e:
                observation = f"Invalid argument for CF recommendation: {argument}. Error: {e}"

        elif action_type.lower() == 'user_similarity':
            # Find similar users
            try:
                if self.json_mode:
                    user_id = argument.get('user_id', argument)
                    top_k = argument.get('top_k', 5)
                else:
                    parts = argument.split(',')
                    user_id = int(parts[0].strip())
                    top_k = int(parts[1].strip()) if len(parts) > 1 else 5

                similar_users = self.cf_tool.find_similar_users(user_id=user_id, top_k=top_k)
                observation = f"Similar users to User {user_id}: {similar_users}"
                log_head = f':violet[Similar Users to] :red[{user_id}]:violet[...]\n- '

            except (ValueError, KeyError) as e:
                observation = f"Invalid argument for user similarity: {argument}. Error: {e}"

        elif action_type.lower() == 'item_similarity':
            # Find similar items
            try:
                if self.json_mode:
                    item_id = argument.get('item_id', argument)
                    top_k = argument.get('top_k', 5)
                else:
                    parts = argument.split(',')
                    item_id = int(parts[0].strip())
                    top_k = int(parts[1].strip()) if len(parts) > 1 else 5

                similar_items = self.cf_tool.find_similar_items(item_id=item_id, top_k=top_k)
                observation = f"Similar items to Item {item_id}: {similar_items}"
                log_head = f':violet[Similar Items to] :red[{item_id}]:violet[...]\n- '

            except (ValueError, KeyError) as e:
                observation = f"Invalid argument for item similarity: {argument}. Error: {e}"

        elif action_type.lower() == 'userinfo':
            # Get user information
            try:
                query_user_id = int(argument)
                observation = self.info_retriever.user_info(user_id=query_user_id)
                log_head = f':violet[UserInfo for] :red[{query_user_id}]:violet[...]\n- '
            except ValueError:
                observation = f"Invalid user id: {argument}"

        elif action_type.lower() == 'iteminfo':
            # Get item information
            try:
                query_item_id = int(argument)
                observation = self.info_retriever.item_info(item_id=query_item_id)
                log_head = f':violet[ItemInfo for] :red[{query_item_id}]:violet[...]\n- '
            except ValueError:
                observation = f"Invalid item id: {argument}"

        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
            log_head = ':violet[Finish with results]:\n- '
        else:
            observation = f'Unknown command type: {action_type}. Valid: recommend, user_similarity, item_similarity, userinfo, iteminfo, finish.'

        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def forward(self, user_id: int, top_k: int = 10, *args, **kwargs) -> str:
        """
        Get collaborative filtering recommendations for a user
        """
        while not self.is_finished():
            command = self._prompt_cf_agent(
                user_id=user_id,
                top_k=top_k
            )
            self.command(command)

        if not self.finished:
            return f'CF Agent did not return recommendations for user {user_id}.'
        return f'CF Recommendations for user {user_id}: {self.results}'

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Handle manager calls for CF recommendations
        """
        if json_mode:
            if not isinstance(argument, dict):
                return f'Invalid argument type for CF agent: {type(argument)}. Expected dict.'
            user_id = argument.get('user_id')
            top_k = argument.get('top_k', self.default_top_k)
        else:
            if not isinstance(argument, str):
                return f'Invalid argument type for CF agent: {type(argument)}. Expected string.'
            # Parse "user_id,top_k" format
            parts = argument.split(',')
            try:
                user_id = int(parts[0].strip())
                top_k = int(parts[1].strip()) if len(parts) > 1 else self.default_top_k
            except (ValueError, IndexError):
                return f'Invalid argument format for CF agent: {argument}. Expected "user_id,top_k".'

        if user_id is None:
            return 'Missing user_id for CF recommendation.'

        return self(user_id=user_id, top_k=top_k)


if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    cf_agent = CollaborativeFilteringAgent(
        config_path='config/agents/collaborative_filtering_agent.json',
        prompts=read_prompts('config/prompts/agent_prompt/collaborative_filtering.json')
    )

    user_id = int(input('User ID: '))
    top_k = int(input('Top K (default 10): ') or '10')
    result = cf_agent(user_id=user_id, top_k=top_k)
    print(result)
