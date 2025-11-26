"""
Ranking Agent
Xếp hạng và chọn top phim từ danh sách candidates
"""

from typing import Any, List, Optional
from loguru import logger
import json

from macrec.agents.base import ToolAgent
from macrec.tools import InfoDatabase, InteractionRetriever, CollaborativeFiltering
from macrec.utils import read_json, get_rm, parse_action


class RankingAgent(ToolAgent):
    """
    Agent xếp hạng và chọn top items từ danh sách candidates
    """
    
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 10)
        self.default_top_k = get_rm(config, 'default_top_k', 10)
        # Remove agent-specific config before passing to LLM
        llm_config = config.copy()
        llm_config.pop('tool_config', None)
        llm_config.pop('max_turns', None)
        llm_config.pop('default_top_k', None)
        self.ranking_agent = self.get_LLM(config=llm_config)
        self.json_mode = self.ranking_agent.json_mode
        self.reset()
    
    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
        }
    
    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']
    
    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']
    
    @property
    def cf_tool(self) -> Optional[CollaborativeFiltering]:
        """Optional Collaborative Filtering tool"""
        return self.tools.get('cf_tool', None)
    
    @property
    def ranking_agent_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['ranking_agent_prompt_json']
        else:
            return self.prompts['ranking_agent_prompt']
    
    @property
    def ranking_agent_examples(self) -> str:
        if self.json_mode:
            return self.prompts['ranking_agent_examples_json']
        else:
            return self.prompts['ranking_agent_examples']
    
    @property
    def hint(self) -> str:
        if 'ranking_agent_hint' not in self.prompts:
            return ''
        return self.prompts['ranking_agent_hint']
    
    def _build_ranking_agent_prompt(self, **kwargs) -> str:
        fewshot_key = 'ranking_agent_fewshot_json' if self.json_mode else 'ranking_agent_fewshot'
        fewshot = self.prompts.get(fewshot_key, '')
        return self.ranking_agent_prompt.format(
            examples=self.ranking_agent_examples,
            ranking_agent_fewshot=fewshot,
            ranking_agent_fewshot_json=fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )
    
    def _prompt_ranking_agent(self, **kwargs) -> str:
        prompt = self._build_ranking_agent_prompt(**kwargs)
        command = self.ranking_agent(prompt)
        return command
    
    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        
        if action_type.lower() == 'userinfo':
            try:
                query_user_id = int(argument)
                observation = self.info_retriever.user_info(user_id=query_user_id)
                log_head = f':violet[Look up UserInfo of user] :red[{query_user_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid user id: {argument}"
        
        elif action_type.lower() == 'iteminfo':
            try:
                query_item_id = int(argument)
                observation = self.info_retriever.item_info(item_id=query_item_id)
                log_head = f':violet[Look up ItemInfo of item] :red[{query_item_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid item id: {argument}"
        
        elif action_type.lower() == 'predictrating':
            # Predict rating for a user-item pair using CF
            if self.cf_tool is not None:
                valid = True
                if self.json_mode:
                    if not isinstance(argument, list) or len(argument) < 2:
                        observation = f"Invalid argument: {argument}. Should be [user_id, item_id] or [user_id, item_id, method, cf_type]"
                        valid = False
                    else:
                        user_id = argument[0]
                        item_id = argument[1]
                        method = argument[2] if len(argument) > 2 else 'pearson'
                        cf_type = argument[3] if len(argument) > 3 else 'user_based'
                        
                        if not isinstance(user_id, int) or not isinstance(item_id, int):
                            observation = f"Invalid user_id or item_id: {user_id}, {item_id}"
                            valid = False
                        elif method not in ['pearson', 'cosine']:
                            observation = f"Invalid method: {method}. Use 'pearson' or 'cosine'"
                            valid = False
                        elif cf_type not in ['user_based', 'item_based']:
                            observation = f"Invalid CF type: {cf_type}. Use 'user_based' or 'item_based'"
                            valid = False
                else:
                    try:
                        parts = argument.split(',')
                        user_id = int(parts[0])
                        item_id = int(parts[1])
                        method = parts[2].strip() if len(parts) > 2 else 'pearson'
                        cf_type = parts[3].strip() if len(parts) > 3 else 'user_based'
                        
                        if method not in ['pearson', 'cosine']:
                            observation = f"Invalid method: {method}. Use 'pearson' or 'cosine'"
                            valid = False
                        elif cf_type not in ['user_based', 'item_based']:
                            observation = f"Invalid CF type: {cf_type}. Use 'user_based' or 'item_based'"
                            valid = False
                    except (ValueError, TypeError) as e:
                        observation = f"Invalid argument format: {argument}. Error: {e}"
                        valid = False
                
                if valid:
                    try:
                        if cf_type == 'user_based':
                            predicted_rating = self.cf_tool.predict_rating_user_based(
                                user_id=user_id,
                                item_id=item_id,
                                method=method,
                                k=50
                            )
                        else:
                            predicted_rating = self.cf_tool.predict_rating_item_based(
                                user_id=user_id,
                                item_id=item_id,
                                method=method,
                                k=50
                            )
                        
                        observation = f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating:.4f}"
                        log_head = f':violet[Predicted rating]:violet[...]\n- '
                    except Exception as e:
                        observation = f"Error predicting rating: {str(e)}"
            else:
                observation = "Collaborative Filtering tool is not configured. Cannot predict rating."
        
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
            log_head = ':violet[Finish with ranked list]:\n- '
        else:
            observation = f'Unknown command type: {action_type}.'
        
        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)
    
    def forward(
        self,
        user_id: int,
        candidate_items: List[int],
        top_k: Optional[int] = None,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Rank candidate items and return top k
        
        Args:
            user_id: ID of the user
            candidate_items: List of candidate item IDs
            top_k: Number of top items to return (default: self.default_top_k)
        """
        if top_k is None:
            top_k = self.default_top_k
        
        if not candidate_items:
            return "No candidate items provided."
        
        # Reset interaction retriever if data sample is available
        if self.system is not None and hasattr(self.system, 'data_sample') and self.system.data_sample is not None:
            if 'user_id' in self.system.data_sample and 'item_id' in self.system.data_sample:
                self.interaction_retriever.reset(
                    user_id=self.system.data_sample['user_id'],
                    item_id=self.system.data_sample['item_id']
                )
        
        # Format candidate items for prompt
        candidate_str = ', '.join(map(str, candidate_items[:50]))  # Limit to 50 for prompt
        if len(candidate_items) > 50:
            candidate_str += f' ... (total {len(candidate_items)} items)'
        
        while not self.is_finished():
            command = self._prompt_ranking_agent(
                user_id=user_id,
                candidate_items=candidate_str,
                n_candidates=len(candidate_items),
                top_k=top_k
            )
            self.command(command)
        
        if not self.finished:
            return "RankingAgent did not return any result."
        
        return self.results
    
    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke ranking agent
        
        Args:
            argument: Can be:
                - list: [user_id, candidate_items] or [user_id, candidate_items, top_k]
                - dict: {'user_id': int, 'candidates': List[int], 'top_k': Optional[int]}
            json_mode: Whether in JSON mode
        """
        if json_mode:
            if isinstance(argument, dict):
                user_id = argument.get('user_id')
                candidate_items = argument.get('candidates', [])
                top_k = argument.get('top_k', None)
            elif isinstance(argument, list) and len(argument) >= 2:
                user_id = argument[0]
                candidate_items = argument[1]
                top_k = argument[2] if len(argument) > 2 else None
            else:
                return f"Invalid argument: {argument}. Should be dict or list [user_id, candidate_items, top_k]"
        else:
            try:
                # Try to parse as JSON first
                if isinstance(argument, str) and argument.strip().startswith('{'):
                    arg_dict = json.loads(argument)
                    user_id = arg_dict.get('user_id')
                    candidate_items = arg_dict.get('candidates', [])
                    top_k = arg_dict.get('top_k', None)
                else:
                    # Parse as comma-separated: user_id,candidate1,candidate2,...,top_k
                    parts = str(argument).split(',')
                    user_id = int(parts[0])
                    candidate_items = [int(x.strip()) for x in parts[1:-1] if x.strip().isdigit()]
                    if parts[-1].strip().isdigit() and len(parts) > 2:
                        # Last part could be top_k or a candidate
                        if len(candidate_items) == 0:
                            # All parts except first are candidates
                            candidate_items = [int(x.strip()) for x in parts[1:] if x.strip().isdigit()]
                            top_k = None
                        else:
                            # Last part is top_k
                            top_k = int(parts[-1])
                    else:
                        top_k = None
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return f"Invalid argument: {argument}. Error: {e}"
        
        if not isinstance(user_id, int):
            return f"Invalid user_id: {user_id}. Must be an integer."
        
        if not isinstance(candidate_items, list) or not all(isinstance(x, int) for x in candidate_items):
            return f"Invalid candidate_items: {candidate_items}. Must be a list of integers."
        
        return self(user_id=user_id, candidate_items=candidate_items, top_k=top_k)

