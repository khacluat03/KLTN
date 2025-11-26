"""
Candidate Generator Agent
Tạo danh sách phim sơ bộ (candidate items) dựa trên:
- User preferences và history
- Similar users
- Popular items
- Collaborative filtering
"""

from typing import Any, List, Optional
from loguru import logger

from macrec.agents.base import ToolAgent
from macrec.tools import InfoDatabase, InteractionRetriever, CollaborativeFiltering
from macrec.utils import read_json, get_rm, parse_action


class CandidateGenerator(ToolAgent):
    """
    Agent tạo danh sách candidate items sơ bộ cho recommendation
    """
    
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 10)
        self.candidate_generator = self.get_LLM(config=config)
        self.json_mode = self.candidate_generator.json_mode
        self.default_candidate_size = get_rm(config, 'default_candidate_size', 50)
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
    def candidate_generator_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['candidate_generator_prompt_json']
        else:
            return self.prompts['candidate_generator_prompt']
    
    @property
    def candidate_generator_examples(self) -> str:
        if self.json_mode:
            return self.prompts['candidate_generator_examples_json']
        else:
            return self.prompts['candidate_generator_examples']
    
    @property
    def hint(self) -> str:
        if 'candidate_generator_hint' not in self.prompts:
            return ''
        return self.prompts['candidate_generator_hint']
    
    def _build_candidate_generator_prompt(self, **kwargs) -> str:
        fewshot_key = 'candidate_generator_fewshot_json' if self.json_mode else 'candidate_generator_fewshot'
        fewshot = self.prompts.get(fewshot_key, '')
        return self.candidate_generator_prompt.format(
            examples=self.candidate_generator_examples,
            candidate_generator_fewshot=fewshot,
            candidate_generator_fewshot_json=fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )
    
    def _prompt_candidate_generator(self, **kwargs) -> str:
        prompt = self._build_candidate_generator_prompt(**kwargs)
        command = self.candidate_generator(prompt)
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
        
        elif action_type.lower() == 'userhistory':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 2:
                    observation = f"Invalid user id and retrieval number: {argument}"
                    valid = False
                else:
                    query_user_id, k = argument
                    if not isinstance(query_user_id, int) or not isinstance(k, int):
                        observation = f"Invalid user id and retrieval number: {argument}"
                        valid = False
            else:
                try:
                    query_user_id, k = argument.split(',')
                    query_user_id = int(query_user_id)
                    k = int(k)
                except (ValueError, TypeError):
                    observation = f"Invalid user id and retrieval number: {argument}"
                    valid = False
            
            if valid:
                observation = self.interaction_retriever.user_retrieve(user_id=query_user_id, k=k)
                log_head = f':violet[Look up UserHistory of user] :red[{query_user_id}] :violet[with at most] :red[{k}] :violet[items...]\n- '
        
        elif action_type.lower() == 'itemhistory':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 2:
                    observation = f"Invalid item id and retrieval number: {argument}"
                    valid = False
                else:
                    query_item_id, k = argument
                    if not isinstance(query_item_id, int) or not isinstance(k, int):
                        observation = f"Invalid item id and retrieval number: {argument}"
                        valid = False
            else:
                try:
                    query_item_id, k = argument.split(',')
                    query_item_id = int(query_item_id)
                    k = int(k)
                except (ValueError, TypeError):
                    observation = f"Invalid item id and retrieval number: {argument}"
                    valid = False
            
            if valid:
                observation = self.interaction_retriever.item_retrieve(item_id=query_item_id, k=k)
                log_head = f':violet[Look up ItemHistory of item] :red[{query_item_id}] :violet[with at most] :red[{k}] :violet[users...]\n- '
        
        elif action_type.lower() == 'generatecandidates':
            # Generate candidates using CF tool if available
            if self.cf_tool is not None:
                valid = True
                if self.json_mode:
                    if not isinstance(argument, list) or len(argument) < 1:
                        observation = f"Invalid argument: {argument}. Should be [user_id] or [user_id, n_items, method]"
                        valid = False
                    else:
                        user_id = argument[0]
                        n_items = argument[1] if len(argument) > 1 else self.default_candidate_size
                        method = argument[2] if len(argument) > 2 else 'pearson'
                        cf_type = argument[3] if len(argument) > 3 else 'user_based'
                        
                        if not isinstance(user_id, int):
                            observation = f"Invalid user id: {user_id}"
                            valid = False
                        elif not isinstance(n_items, int) or n_items <= 0:
                            observation = f"Invalid number of items: {n_items}"
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
                        n_items = int(parts[1]) if len(parts) > 1 else self.default_candidate_size
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
                            recommendations = self.cf_tool.recommend_items_user_based(
                                user_id=user_id,
                                n_items=n_items,
                                method=method,
                                k=50
                            )
                        else:
                            recommendations = self.cf_tool.recommend_items_item_based(
                                user_id=user_id,
                                n_items=n_items,
                                method=method,
                                k=50
                            )
                        
                        # Format as list of item IDs
                        candidate_items = [item_id for item_id, _ in recommendations]
                        observation = f"Generated {len(candidate_items)} candidate items for user {user_id}: {candidate_items[:20]}{'...' if len(candidate_items) > 20 else ''}"
                        log_head = f':violet[Generated candidates for user] :red[{user_id}]:violet[...]\n- '
                    except Exception as e:
                        observation = f"Error generating candidates: {str(e)}"
            else:
                observation = "Collaborative Filtering tool is not configured. Cannot generate candidates using CF."
        
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
            log_head = ':violet[Finish with candidate list]:\n- '
        else:
            observation = f'Unknown command type: {action_type}.'
        
        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)
    
    def forward(self, user_id: int, n_candidates: Optional[int] = None, *args: Any, **kwargs: Any) -> str:
        """
        Generate candidate items for a user
        
        Args:
            user_id: ID of the user
            n_candidates: Number of candidates to generate (default: self.default_candidate_size)
        """
        if n_candidates is None:
            n_candidates = self.default_candidate_size
        
        # Reset interaction retriever if data sample is available
        if self.system is not None and hasattr(self.system, 'data_sample') and self.system.data_sample is not None:
            if 'user_id' in self.system.data_sample and 'item_id' in self.system.data_sample:
                self.interaction_retriever.reset(
                    user_id=self.system.data_sample['user_id'],
                    item_id=self.system.data_sample['item_id']
                )
        
        while not self.is_finished():
            command = self._prompt_candidate_generator(
                user_id=user_id,
                n_candidates=n_candidates
            )
            self.command(command)
        
        if not self.finished:
            return "CandidateGenerator did not return any result."
        
        return self.results
    
    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke candidate generator
        
        Args:
            argument: Can be:
                - int: user_id (uses default n_candidates)
                - list: [user_id] or [user_id, n_candidates]
            json_mode: Whether in JSON mode
        """
        if json_mode:
            if isinstance(argument, int):
                user_id = argument
                n_candidates = None
            elif isinstance(argument, list) and len(argument) >= 1:
                user_id = argument[0]
                n_candidates = argument[1] if len(argument) > 1 else None
            else:
                return f"Invalid argument: {argument}. Should be int (user_id) or list [user_id, n_candidates]"
        else:
            try:
                parts = str(argument).split(',')
                user_id = int(parts[0])
                n_candidates = int(parts[1]) if len(parts) > 1 else None
            except (ValueError, TypeError):
                return f"Invalid argument: {argument}. Should be 'user_id' or 'user_id,n_candidates'"
        
        if not isinstance(user_id, int):
            return f"Invalid user_id: {user_id}. Must be an integer."
        
        return self(user_id=user_id, n_candidates=n_candidates)

