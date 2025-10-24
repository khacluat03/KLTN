from typing import Any
from loguru import logger

from macrec.agents.base import ToolAgent
from macrec.tools import InfoDatabase, InteractionRetriever
from macrec.utils import read_json, get_rm, parse_action

class PersonalizationAgent(ToolAgent):
    """
    The PersonalizationAgent analyzes user preferences and creates personalized recommendations
    based on user profile, interaction history, and contextual information.
    """
    
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 15)
        self.personalization_llm = self.get_LLM(config=config)
        self.json_mode = self.personalization_llm.json_mode
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
    def personalization_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['personalization_prompt_json']
        else:
            return self.prompts['personalization_prompt']

    @property
    def personalization_examples(self) -> str:
        if self.json_mode:
            return self.prompts['personalization_examples_json']
        else:
            return self.prompts['personalization_examples']

    @property
    def personalization_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['personalization_fewshot_json']
        else:
            return self.prompts['personalization_fewshot']

    @property
    def personalization_hint(self) -> str:
        return self.prompts['personalization_hint']

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke the PersonalizationAgent to create personalized recommendations.
        
        Args:
            argument: The user ID or user context to personalize for
            json_mode: Whether to use JSON format for responses
            
        Returns:
            Personalized recommendation analysis
        """
        self.observation(f'Starting personalization analysis for: {argument}')
        
        # Build the prompt
        prompt = self.personalization_prompt.format(
            user_id=argument,
            task_type=self.system.task_type if self.system else 'recommendation',
            max_step=self.max_turns,
            examples=self.personalization_examples,
            fewshot=self.personalization_fewshot,
            history=self.history,
            hint=self.personalization_hint
        )
        
        # Get response from LLM
        response = self.personalization_llm(prompt)
        self.observation(f'PersonalizationAgent response: {response}')
        
        # Parse and execute the response
        if json_mode:
            try:
                import json
                action = json.loads(response)
                action_type = action.get('type', '')
                content = action.get('content', '')
            except:
                action_type, content = parse_action(response, json_mode=False)
        else:
            action_type, content = parse_action(response, json_mode=False)
        
        # Execute the action
        observation = self._execute_action(action_type, content)
        self._history.append((response, observation))
        
        # Check if finished
        if action_type.lower() == 'finish':
            return self.finish(content)
        
        # Continue with next iteration
        return self.invoke(argument, json_mode)

    def _execute_action(self, action_type: str, content: Any) -> str:
        """Execute the action returned by the LLM."""
        try:
            if action_type.lower() == 'userinfo':
                user_id = content if isinstance(content, (int, str)) else content[0]
                result = self.info_retriever.lookup('user', str(user_id))
                self.observation(f'Retrieved user info for {user_id}')
                return result
                
            elif action_type.lower() == 'iteminfo':
                item_id = content if isinstance(content, (int, str)) else content[0]
                result = self.info_retriever.lookup('item', str(item_id))
                self.observation(f'Retrieved item info for {item_id}')
                return result
                
            elif action_type.lower() == 'userhistory':
                if isinstance(content, list) and len(content) >= 2:
                    user_id, k = content[0], content[1]
                else:
                    user_id, k = content, 5  # default to 5 interactions
                result = self.interaction_retriever.search(f'user:{user_id}', k)
                self.observation(f'Retrieved {k} interactions for user {user_id}')
                return result
                
            elif action_type.lower() == 'itemhistory':
                if isinstance(content, list) and len(content) >= 2:
                    item_id, k = content[0], content[1]
                else:
                    item_id, k = content, 5  # default to 5 interactions
                result = self.interaction_retriever.search(f'item:{item_id}', k)
                self.observation(f'Retrieved {k} interactions for item {item_id}')
                return result
                
            elif action_type.lower() == 'finish':
                self.observation(f'Personalization analysis completed: {content}')
                return f'Personalization completed: {content}'
                
            else:
                error_msg = f'Unknown action type: {action_type}'
                self.observation(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f'Error executing action {action_type}: {str(e)}'
            self.observation(error_msg)
            return error_msg

    def forward(self, user_id: str, context: str = "", *args, **kwargs) -> str:
        """
        Forward pass for personalization.
        
        Args:
            user_id: The user ID to personalize for
            context: Additional context for personalization
            
        Returns:
            Personalized recommendation analysis
        """
        self.reset()
        return self.invoke(user_id, self.json_mode)
