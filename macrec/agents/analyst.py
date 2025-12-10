from typing import Any
from loguru import logger
import ast

from macrec.agents.base import ToolAgent
from macrec.tools.info_database import InfoDatabase
from macrec.tools.interaction import InteractionRetriever
from macrec.tools.rating_predictor import RatingPredictor
from macrec.tools.sequential_predictor import SequentialPredictor
from macrec.utils import read_json, get_rm, parse_action

class Analyst(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 20)
        self.analyst = self.get_LLM(config=config)
        self.json_mode = self.analyst.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
            'rating_predictor': RatingPredictor,
            'sequential_predictor': SequentialPredictor,
        }

    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']

    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']

    @property
    def rating_predictor(self) -> RatingPredictor:
        return self.tools['rating_predictor']
    
    @property
    def sequential_predictor(self) -> SequentialPredictor:
        return self.tools['sequential_predictor']

    @property
    def analyst_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_prompt_json']
        else:
            return self.prompts['analyst_prompt']

    @property
    def analyst_examples(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_examples_json']
        else:
            return self.prompts['analyst_examples']

    @property
    def analyst_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_fewshot_json']
        else:
            return self.prompts['analyst_fewshot']

    @property
    def hint(self) -> str:
        if 'analyst_hint' not in self.prompts:
            return ''
        return self.prompts['analyst_hint']

    def _build_analyst_prompt(self, **kwargs) -> str:
        # Ensure all required placeholders are available
        if 'task_type' not in kwargs:
            kwargs['task_type'] = 'recommendation'  # Default fallback for analyst tasks
        if 'user_id' not in kwargs or kwargs['user_id'] is None:
            kwargs['user_id'] = 'None'
        if 'item_id' not in kwargs or kwargs['item_id'] is None:
            kwargs['item_id'] = 'None'
        return self.analyst_prompt.format(
            examples=self.analyst_examples,
            fewshot=self.analyst_fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )

    def _prompt_analyst(self, **kwargs) -> str:
        analyst_prompt = self._build_analyst_prompt(**kwargs)
        command = self.analyst(analyst_prompt)
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
            except ValueError or TypeError:
                observation = f"Invalid user id: {argument}"
        elif action_type.lower() == 'iteminfo':
            try:
                if isinstance(argument, list):
                    # Batch processing for list of item IDs
                    item_ids = [int(x) for x in argument]
                    observations = []
                    for item_id in item_ids:
                        info = self.info_retriever.item_info(item_id=item_id)
                        observations.append(f"{item_id}: {info}")
                    observation = "\n".join(observations)
                    log_head = f':violet[Look up ItemInfo for items] :red[{item_ids}]:violet[...]\n- '
                else:
                    # Single item processing
                    query_item_id = int(argument)
                    observation = self.info_retriever.item_info(item_id=query_item_id)
                    log_head = f':violet[Look up ItemInfo of item] :red[{query_item_id}]:violet[...]\n- '
            except (ValueError, TypeError) as e:
                observation = f"Invalid item id: {argument}. Error: {e}"
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
                except ValueError or TypeError:
                    observation = f"Invalid user id and retrieval number: {argument}"
                    valid = False
            if valid:
                # Enforce minimum k=10 for UserHistory to ensure sufficient data for analysis
                if k < 10:
                    logger.warning(f"Analyst requested k={k} for UserHistory. Enforcing minimum k=10.")
                    k = 10
                observation = self.interaction_retriever.user_retrieve(user_id=query_user_id, k=k)
                log_head = f':violet[Look up UserHistory of user] :red[{query_user_id}] :violet[with at most] :red[{k}] :violet[items...]\n- '
        elif action_type.lower() == 'itemhistory':
            valid = True
            if self.json_mode:
                # Check for batch processing format: [[id1, id2, ...], k]
                if isinstance(argument, list) and len(argument) == 2 and isinstance(argument[0], list):
                    item_ids, k = argument
                    observations = []
                    for item_id in item_ids:
                         try:
                             info = self.interaction_retriever.item_retrieve(item_id=int(item_id), k=k)
                             observations.append(f"Item {item_id}: {info}")
                         except Exception as e:
                             observations.append(f"Item {item_id}: Error {e}")
                    observation = "\n".join(observations)
                    log_head = f':violet[Look up ItemHistory for items] :red[{item_ids}] :violet[with at most] :red[{k}] :violet[users...]\n- '
                elif not isinstance(argument, list) or len(argument) != 2:
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
                except ValueError or TypeError:
                    observation = f"Invalid item id and retrieval number: {argument}"
                    valid = False
            if valid and not (self.json_mode and isinstance(argument, list) and len(argument) == 2 and isinstance(argument[0], list)):
                observation = self.interaction_retriever.item_retrieve(item_id=query_item_id, k=k)
                log_head = f':violet[Look up ItemHistory of item] :red[{query_item_id}] :violet[with at most] :red[{k}] :violet[users...]\n- '
        elif action_type.lower() == 'predictrating':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 2:
                    observation = f"Invalid user id and item id: {argument}"
                    valid = False
                else:
                    query_user_id, query_item_id = argument
                    if not isinstance(query_user_id, int) or not isinstance(query_item_id, int):
                        observation = f"Invalid user id and item id: {argument}"
                        valid = False
            else:
                try:
                    query_user_id, query_item_id = argument.split(',')
                    query_user_id = int(query_user_id)
                    query_item_id = int(query_item_id)
                except ValueError or TypeError:
                    observation = f"Invalid user id and item id: {argument}"
                    valid = False
            if valid:
                observation = self.rating_predictor.predict(user_id=query_user_id, item_id=query_item_id)
                log_head = f':violet[Predict Rating for user] :red[{query_user_id}] :violet[and item] :red[{query_item_id}]:violet[...]\n- '
        # elif action_type.lower() == 'predictnext':
        #     # ⚠️ DEPRECATED: PredictNext is no longer used in Analyst prompt
        #     # Manager now calls SequentialRecommendationAgent directly for sequential recommendations
        #     # This code is kept for backward compatibility only
        #     # See: SequentialRecommendationAgent in macrec/agents/sequential_recommendation_agent.py
            
        #     # Predict next items for a user using SASRec
        #     try:
        #         query_user_id = int(argument)
        #         top_items = self.sequential_predictor.predict_next(user_id=query_user_id, top_k=5)
        #         if top_items:
        #             # Get item info for top items
        #             item_names = []
        #             for item_id in top_items:
        #                 item_info = self.info_retriever.item_info(item_id=item_id)
        #                 item_names.append(f"{item_id}: {item_info}")
        #             observation = f"Top 5 next items for user {query_user_id}:\n" + "\n".join(item_names)
        #         else:
        #             observation = f"No sequential predictions available for user {query_user_id}"
        #         log_head = f':violet[Predict Next Items for user] :red[{query_user_id}]:violet[...]\n- '
        #     except ValueError or TypeError:
        #         observation = f"Invalid user id: {argument}"
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
            log_head = ':violet[Finish with results]:\n- '
        else:
            observation = f'Unknown command type: {action_type}.'
        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def forward(self, id: int, analyse_type: str, *args: Any, **kwargs: Any) -> str:
        # Handle missing data_sample (e.g. in chat mode)
        user_id = None
        item_id = None
        if hasattr(self.system, 'data_sample') and self.system.data_sample is not None:
            user_id = self.system.data_sample.get('user_id')
            item_id = self.system.data_sample.get('item_id')
        
        # Override with explicit arguments if provided (e.g. from chat)
        if analyse_type == 'user':
            user_id = id
        elif analyse_type == 'item':
            item_id = id
        
        # Check if extra kwargs provided user_id/item_id
        if 'user_id' in kwargs: user_id = kwargs['user_id']
        if 'item_id' in kwargs: item_id = kwargs['item_id']
        
        # If user_id/item_id are not in data_sample, they might be passed via kwargs or inferred later
        # For now, we pass what we have. The prompt context will show "None" if missing.
        
        # For rating prediction (both user_id and item_id provided), skip interaction reset
        # because the interaction may not exist yet (that's what we're predicting!)
        if user_id is not None and item_id is not None:
             # Skip reset - let PredictRating handle this directly
             pass
        else:
             # In chat mode or general analysis, reset with available context
             self.interaction_retriever.reset()
             
        while not self.is_finished():
            command = self._prompt_analyst(
                id=id, 
                analyse_type=analyse_type,
                user_id=user_id,
                item_id=item_id
            )
            self.command(command)
        if not self.finished:
            return "Analyst did not return any result."
        return self.results

    def invoke(self, argument: Any, json_mode: bool) -> str:
        # Handle 4-argument case for rating prediction FIRST: Analyse[user, 123, item, 456]
        user_id = None
        item_id = None
        
        if json_mode:
             if isinstance(argument, list) and len(argument) == 4:
                 if argument[0] == 'user' and argument[2] == 'item':
                     analyse_type = 'user' # Primary type
                     id = argument[1]
                     user_id = argument[1]
                     item_id = argument[3]
                     return self(analyse_type=analyse_type, id=id, user_id=user_id, item_id=item_id)
        else:
             parts = argument.split(',')
             if len(parts) == 4:
                 t1, id1, t2, id2 = [p.strip() for p in parts]
                 if t1 == 'user' and t2 == 'item':
                     try:
                         return self(analyse_type='user', id=int(id1), user_id=int(id1), item_id=int(id2))
                     except ValueError:
                         pass
        
        # Handle standard 2-argument case: Analyse[user, 123] or Analyse[item, 456]
        if json_mode:
            if not isinstance(argument, list) or len(argument) != 2:
                observation = "The argument of the action 'Analyse' should be a list with two elements: analyse type (user or item) and id, OR four elements for rating prediction: [user, user_id, item, item_id]."
                return observation
            else:
                analyse_type, id = argument
                if (isinstance(id, str) and 'user_' in id) or (isinstance(id, str) and 'item_' in id):
                    observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    return observation
                elif analyse_type.lower() not in ['user', 'item']:
                    observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                    return observation
                elif not isinstance(id, int):
                    observation = f"Invalid id: {id}. It should be an integer."
                    return observation
        else:
            # Try to parse list argument first for batch processing
            try:
                # Check for pattern "type, [list]"
                parts = argument.split(',', 1)
                if len(parts) == 2:
                    analyse_type = parts[0].strip()
                    potential_list = parts[1].strip()
                    if potential_list.startswith('[') and potential_list.endswith(']'):
                        id_list = ast.literal_eval(potential_list)
                        if isinstance(id_list, list):
                             return self(analyse_type=analyse_type, id=id_list)
            except (ValueError, SyntaxError):
                pass

            if len(argument.split(',')) != 2:
                observation = "The argument of the action 'Analyse' should be a string with two elements separated by a comma: analyse type (user or item) and id, OR four elements for rating prediction: user, user_id, item, item_id. For batch analysis, use: type, [id1, id2, ...]"
                return observation
            else:
                analyse_type, id = argument.split(',')
                if 'user_' in id or 'item_' in id:
                    observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    return observation
                elif analyse_type.lower() not in ['user', 'item']:
                    observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                    return observation
                else:
                    try:
                        id = int(id)
                    except ValueError or TypeError:
                        observation = f"Invalid id: {id}. The id should be an integer."
                        return observation

        return self(analyse_type=analyse_type, id=id)

if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from macrec.utils import init_openai_api, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    prompts = read_prompts('config/prompts/old_system_prompt/react_analyst.json')
    for prompt_name, prompt_template in prompts.items():
        if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
            prompts[prompt_name] = prompt_template.partial(task_type='rating prediction')
    analyst = Analyst(config_path='config/agents/analyst_Beauty.json', prompts=prompts)
    user_id, item_id = list(map(int, input('User id and item id: ').split()))
    result = analyst(user_id=user_id, item_id=item_id)
