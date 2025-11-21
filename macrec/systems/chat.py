from typing import Any
from loguru import logger

from macrec.systems.base import System
from macrec.agents import Manager, Searcher, Interpreter
from macrec.utils import format_chat_history, parse_action

class ChatSystem(System):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['chat']

    def init(self, *args, **kwargs) -> None:
        self.manager = Manager(thought_config_path=self.config['manager_thought'], action_config_path=self.config['manager_action'], **self.agent_kwargs)
        self.searcher = Searcher(config_path=self.config['searcher'], **self.agent_kwargs)
        self.interpreter = Interpreter(config_path=self.config['interpreter'], **self.agent_kwargs)
        self.max_step: int = self.config.get('max_step', 6)
        self.require_search_before_finish: bool = self.config.get('require_search_before_finish', True)
        self.manager_kwargs = {
            "max_step": self.max_step,
        }

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(history=self.chat_history, task_prompt=self.task_prompt, scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        if clear:
            self._chat_history = []
        self._reset_action_history()
        self.searcher.reset()
        self.interpreter.reset()

    def _reset_action_history(self) -> None:
        self.step_n: int = 1
        self._search_performed: bool = False
        self.action_history = []
        self._is_database_task: bool = False
    
    def _detect_database_task(self, text: str) -> bool:
        """Detect if task is related to database queries"""
        if not text:
            return False
        text_lower = text.lower()
        db_keywords = [
            'user', 'item', 'interaction', 'rating', 'phim', 'movie', 'film',
            'xem', 'watch', 'đã xem', 'watched', 'đánh giá', 'review',
            'database', 'bảng', 'table', 'truy vấn', 'query', 'sql',
            'select', 'top', 'liệt kê', 'tìm', 'find'
        ]
        return any(keyword in text_lower for keyword in db_keywords)

    def add_chat_history(self, chat: str, role: str) -> None:
        self._chat_history.append((chat, role))

    @property
    def chat_history(self) -> list[tuple[str, str]]:
        return format_chat_history(self._chat_history)

    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(history=self.chat_history, task_prompt=self.task_prompt, scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)

    def act(self) -> tuple[str, Any]:
        # Act
        # Detect if this is a database-related task
        if not hasattr(self, '_is_database_task') or self.step_n == 1:
            self._is_database_task = self._detect_database_task(getattr(self, 'task_prompt', ''))
        
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        if self.step_n == self.max_step:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        
        # Strong reminder for database tasks
        if self._is_database_task and not self._search_performed:
            self.scratchpad += '\n⚠️ CRITICAL: This task requires querying the database. You MUST use Search[...] action with a natural language query. You CANNOT Finish without searching first. Example: Search[top 5 users who watched movie Twister] or Search[users who watched movie Copycat]'
        
        # Check if previous thought mentioned Search but haven't searched yet
        if self.require_search_before_finish and not self._search_performed:
            last_thought = self.scratchpad.split('Thought')[-1].split('Action')[0].lower() if 'Thought' in self.scratchpad else ''
            if 'search' in last_thought and 'finish' not in last_thought:
                # Thought mentioned search, remind to use Search action
                self.scratchpad += '\nREMINDER: Your Thought mentioned searching. You MUST use Search[...] action now, not Finish. You cannot Finish until you have executed at least one Search action.'
        
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(history=self.chat_history, task_prompt=self.task_prompt, scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        
        # Auto-convert Finish to Search for database tasks
        if action_type.lower() == 'finish' and self._is_database_task and not self._search_performed:
            logger.warning(f'Auto-converting Finish to Search for database task. Original: {action}')
            # Generate a suggested SQL query based on task
            suggested_query = self._generate_sql_suggestion(getattr(self, 'task_prompt', ''))
            action_type = 'search'
            argument = suggested_query
            action = f'Search[{suggested_query}]'
            self.scratchpad += '\n⚠️ AUTO-CORRECTED: Finish action was converted to Search action because this is a database query task.'
        
        logger.debug(f'Action {self.step_n}: {action}')
        return action_type, argument
    
    def _generate_sql_suggestion(self, task_prompt: str) -> str:
        """Generate a suggested natural language query based on task prompt"""
        # Return the task prompt as-is, as it's already in natural language
        # The RAG SQL tool will handle SQL generation
        return task_prompt

    def execute(self, action_type: str, argument: Any):
        # Execute
        log_head = ''
        if action_type.lower() == 'finish':
            if self.require_search_before_finish and not self._search_performed:
                # Provide helpful guidance when Finish is blocked
                task_prompt_lower = getattr(self, 'task_prompt', '').lower()
                if any(keyword in task_prompt_lower for keyword in ['sql', 'query', 'database', 'select', 'user', 'item', 'interaction']):
                    observation = 'Finish action blocked: you must call Search[...] first. Based on the task, you should use Search[<natural language query>] to query the database. Example: Search[top 5 users who watched movie Twister] or Search[users who watched movie Copycat]'
                else:
                    observation = 'Finish action blocked: you must call Search[...] at least once before finishing. Use Search[<your query>] to search for information first.'
                log_head = ':violet[Finish blocked - must Search first]:\n- '
            else:
                observation = self.finish(argument)
                log_head = ':violet[Finish with results]:\n- '
        elif action_type.lower() == 'search':
            self.log(f':violet[Calling] :red[Searcher] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
            observation = self.searcher.invoke(argument=argument, json_mode=self.manager.json_mode)
            log_head = f':violet[Response from] :red[Searcher] :violet[with] :blue[{argument}]:violet[:]\n- '
            self._search_performed = True
        else:
            observation = f'Invalid Action type or format: {action_type}. Valid Action examples are {self.manager.valid_action_example}.'
        self.scratchpad += f'\nObservation: {observation}'

        logger.debug(f'Observation: {observation}')
        self.log(f'{log_head}{observation}', agent=self.manager, logging=False)

    def step(self) -> None:
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1

    def forward(self, user_input: str, reset: bool = True) -> str:
        if reset:
            self.reset()
        self.add_chat_history(user_input, role='user')
        self.task_prompt = self.interpreter(input=self.chat_history)
        while not self.is_finished() and not self.is_halted():
            self.step()
        if not self.is_finished():
            self.answer = "I'm sorry, I cannot continue the conversation. Please try again."
        self.add_chat_history(self.answer, role='system')
        return self.answer

    def chat(self) -> None:
        print("Start chatting with the system. Type 'exit' or 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = self(user_input, reset=True)
            print(f"System: {response}")

if __name__ == "__main__":
    from macrec.utils import init_openai_api, read_json
    init_openai_api(read_json('config/api-config.json'))
    chat_system = ChatSystem(config_path='config/systems/chat/config.json', task='chat')
    chat_system.chat()
# 1. Hello! How are you today?
# 2. I have watched the movie Schindler's List recently. I am very touched by the movie. I wonder what other movies can teach me about history like this?
