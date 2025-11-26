"""
Sequential Recommendation Agent
Đặc biệt dành cho Sequential/Pattern-based recommendations
"""

from typing import Any, List, Optional
from loguru import logger
import json

from macrec.agents.base import ToolAgent
from macrec.tools.info_database import InfoDatabase
from macrec.tools.interaction import InteractionRetriever
from macrec.tools.sequential_predictor import SequentialPredictor
from macrec.utils import read_json, get_rm, parse_action


class SequentialRecommendationAgent(ToolAgent):
    """
    Agent chuyên về Sequential Recommendations
    Sử dụng temporal patterns và user history để predict next items
    """

    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 8)
        self.default_top_k = get_rm(config, 'default_top_k', 5)
        # Remove agent-specific config before passing to LLM
        llm_config = config.copy()
        llm_config.pop('tool_config', None)
        llm_config.pop('max_turns', None)
        llm_config.pop('default_top_k', None)
        self.seq_agent = self.get_LLM(config=llm_config)
        self.json_mode = self.seq_agent.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
            'seq_tool': SequentialPredictor,
        }

    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']

    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']

    @property
    def seq_tool(self) -> SequentialPredictor:
        return self.tools['seq_tool']

    @property
    def seq_agent_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['seq_agent_prompt_json']
        else:
            return self.prompts['seq_agent_prompt']

    @property
    def seq_agent_examples(self) -> str:
        if self.json_mode:
            return self.prompts['seq_agent_examples_json']
        else:
            return self.prompts['seq_agent_examples']

    @property
    def seq_agent_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['seq_agent_fewshot_json']
        else:
            return self.prompts['seq_agent_fewshot']

    @property
    def hint(self) -> str:
        if 'seq_agent_hint' not in self.prompts:
            return ''
        return self.prompts['seq_agent_hint']

    def _build_seq_agent_prompt(self, **kwargs) -> str:
        return self.seq_agent_prompt.format(
            examples=self.seq_agent_examples,
            fewshot=self.seq_agent_fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )

    def _prompt_seq_agent(self, **kwargs) -> str:
        seq_agent_prompt = self._build_seq_agent_prompt(**kwargs)
        command = self.seq_agent(seq_agent_prompt)
        return command

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)

        if action_type.lower() == 'predict_next' or action_type.lower() == 'sequential':
            # Sequential prediction - predict next items
            try:
                if self.json_mode:
                    user_id = argument.get('user_id', argument)
                    top_k = argument.get('top_k', self.default_top_k)
                else:
                    # Parse "user_id,top_k" format
                    parts = argument.split(',')
                    user_id = int(parts[0].strip())
                    top_k = int(parts[1].strip()) if len(parts) > 1 else self.default_top_k

                # Get sequential predictions
                next_items = self.seq_tool.predict_next(user_id=user_id, top_k=top_k)

                # Get item details for better presentation
                if next_items:
                    detailed_preds = []
                    for item_id in next_items:
                        item_info = self.info_retriever.item_info(item_id=item_id)
                        detailed_preds.append(f"{item_id}: {item_info}")
                    observation = f"Sequential Predictions for User {user_id} (next {top_k}):\n" + "\n".join(detailed_preds)
                else:
                    observation = f"No sequential predictions available for user {user_id}"

                log_head = f':violet[Sequential Prediction for User] :red[{user_id}]:violet[...]\n- '

            except (ValueError, KeyError, IndexError) as e:
                observation = f"Invalid argument for sequential prediction: {argument}. Error: {e}"

        elif action_type.lower() == 'user_history':
            # Get detailed user interaction history for sequential analysis
            try:
                if self.json_mode:
                    user_id = argument.get('user_id', argument)
                    limit = argument.get('limit', 20)
                else:
                    parts = argument.split(',')
                    user_id = int(parts[0].strip())
                    limit = int(parts[1].strip()) if len(parts) > 1 else 20

                # Get recent user history
                history_data = self.interaction_retriever.user_retrieve(user_id=user_id, k=limit)

                # Format as chronological sequence
                if history_data:
                    # Assuming history_data contains timestamps, sort by time
                    observation = f"User {user_id} interaction sequence (last {limit}):\n"
                    for i, (item_id, rating, timestamp) in enumerate(history_data):
                        item_info = self.info_retriever.item_info(item_id=item_id)
                        observation += f"{i+1}. {item_id}: {item_info} (rating: {rating}, time: {timestamp})\n"
                else:
                    observation = f"No interaction history found for user {user_id}"

                log_head = f':violet[User History for] :red[{user_id}]:violet[...]\n- '

            except (ValueError, KeyError) as e:
                observation = f"Invalid argument for user history: {argument}. Error: {e}"

        elif action_type.lower() == 'pattern_analysis':
            # Analyze user behavior patterns
            try:
                if self.json_mode:
                    user_id = argument.get('user_id', argument)
                else:
                    user_id = int(argument)

                # Get user history and analyze patterns
                history_data = self.interaction_retriever.user_retrieve(user_id=user_id, k=50)

                if history_data:
                    # Simple pattern analysis
                    ratings = [rating for _, rating, _ in history_data]
                    avg_rating = sum(ratings) / len(ratings)

                    # Genre preferences over time
                    genre_counts = {}
                    for item_id, _, _ in history_data[-20:]:  # Last 20 interactions
                        item_info = self.info_retriever.item_info(item_id=item_id)
                        # Extract genre (simple parsing)
                        if 'genre' in item_info.lower():
                            genre = item_info.split('Genres:')[1].split('|')[0] if 'Genres:' in item_info else 'Unknown'
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1

                    top_genre = max(genre_counts.items(), key=lambda x: x[1]) if genre_counts else ('Unknown', 0)

                    observation = f"Pattern Analysis for User {user_id}:\n"
                    observation += f"- Average Rating: {avg_rating:.2f}\n"
                    observation += f"- Recent Genre Preference: {top_genre[0]} ({top_genre[1]} items)\n"
                    observation += f"- Total Interactions: {len(history_data)}"
                else:
                    observation = f"No data available for pattern analysis of user {user_id}"

                log_head = f':violet[Pattern Analysis for User] :red[{user_id}]:violet[...]\n- '

            except (ValueError, KeyError) as e:
                observation = f"Invalid argument for pattern analysis: {argument}. Error: {e}"

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
            observation = f'Unknown command type: {action_type}. Valid: predict_next, user_history, pattern_analysis, userinfo, iteminfo, finish.'

        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def forward(self, user_id: int, top_k: int = 5, *args, **kwargs) -> str:
        """
        Get sequential recommendations for a user
        """
        while not self.is_finished():
            command = self._prompt_seq_agent(
                user_id=user_id,
                top_k=top_k
            )
            self.command(command)

        if not self.finished:
            return f'Sequential Agent did not return predictions for user {user_id}.'
        return f'Sequential Predictions for user {user_id}: {self.results}'

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Handle manager calls for sequential recommendations
        """
        if json_mode:
            if not isinstance(argument, dict):
                return f'Invalid argument type for sequential agent: {type(argument)}. Expected dict.'
            user_id = argument.get('user_id')
            top_k = argument.get('top_k', self.default_top_k)
        else:
            if not isinstance(argument, str):
                return f'Invalid argument type for sequential agent: {type(argument)}. Expected string.'
            # Parse "user_id,top_k" format
            parts = argument.split(',')
            try:
                user_id = int(parts[0].strip())
                top_k = int(parts[1].strip()) if len(parts) > 1 else self.default_top_k
            except (ValueError, IndexError):
                return f'Invalid argument format for sequential agent: {argument}. Expected "user_id,top_k".'

        if user_id is None:
            return 'Missing user_id for sequential recommendation.'

        return self(user_id=user_id, top_k=top_k)


if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    seq_agent = SequentialRecommendationAgent(
        config_path='config/agents/sequential_recommendation_agent.json',
        prompts=read_prompts('config/prompts/agent_prompt/sequential_recommendation.json')
    )

    user_id = int(input('User ID: '))
    top_k = int(input('Top K (default 5): ') or '5')
    result = seq_agent(user_id=user_id, top_k=top_k)
    print(result)
