from typing import Any, Optional
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.base import ToolAgent
from macrec.tools import Wikipedia, RAGTextToSQL
from macrec.utils import read_json, parse_action, get_rm

class Searcher(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 6)
        self.searcher = self.get_LLM(config=config)
        self.json_mode = self.searcher.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'retriever': Wikipedia,
        }

    @property
    def retriever(self) -> Wikipedia:
        return self.tools['retriever']
    
    @property
    def sql_tool(self) -> Optional[RAGTextToSQL]:
        """Optional RAG SQL tool"""
        return self.tools.get('sql_tool', None)

    @property
    def searcher_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['searcher_prompt_json']
        else:
            return self.prompts['searcher_prompt']

    @property
    def searcher_examples(self) -> str:
        if self.json_mode:
            return self.prompts['searcher_examples_json']
        else:
            return self.prompts['searcher_examples']

    @property
    def hint(self) -> str:
        if 'searcher_hint' not in self.prompts:
            return ''
        return self.prompts['searcher_hint']

    def _build_searcher_prompt(self, **kwargs) -> str:
        return self.searcher_prompt.format(
            examples=self.searcher_examples,
            k=self.retriever.top_k,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )

    def _prompt_searcher(self, **kwargs) -> str:
        searcher_prompt = self._build_searcher_prompt(**kwargs)
        command = self.searcher(searcher_prompt)
        return command

    def _is_database_query(self, query: str) -> bool:
        """Check if query is related to database (users, items, interactions, movies, etc.)"""
        if not isinstance(query, str):
            return False
        query_lower = query.lower()
        # Check for SQL keywords
        sql_keywords = ['select', 'from', 'where', 'join', 'user', 'item', 'interaction', 'rating']
        if any(keyword in query_lower for keyword in sql_keywords):
            return True
        # Check for database-related terms (Vietnamese and English)
        db_keywords = [
            'user', 'item', 'interaction', 'rating', 'phim', 'movie', 'film',
            'xem', 'watch', 'đã xem', 'watched', 'đánh giá', 'review',
            'database', 'bảng', 'table', 'truy vấn', 'query'
        ]
        return any(keyword in query_lower for keyword in db_keywords)
    
    def _is_sql_query(self, text: str) -> bool:
        """Check if text is a SQL query"""
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        if not stripped:
            return False
        sql_prefixes = ('select', 'with', 'pragma')
        return stripped.lower().startswith(sql_prefixes)

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'search':
            # Prioritize SQL tool if query is database-related
            if self.sql_tool is not None:
                # Check if argument is SQL or database-related
                if self._is_sql_query(argument) or self._is_database_query(argument):
                    logger.debug(f'Using SQL tool for database query: {argument}')
                    # Reset SQL tool and disable cache to ensure fresh results
                    self.sql_tool.reset()
                    observation = self.sql_tool.query(query=argument, use_cache=False)
                    log_head = f':violet[SQL Query] :red[{argument}]:violet[...]\n- '
                else:
                    # Use Wikipedia for general queries
                    observation = self.retriever.search(query=argument)
                    log_head = f':violet[Search for] :red[{argument}]:violet[...]\n- '
            else:
                # No SQL tool, use Wikipedia
                observation = self.retriever.search(query=argument)
                log_head = f':violet[Search for] :red[{argument}]:violet[...]\n- '
        elif action_type.lower() == 'sql' or action_type.lower() == 'query':
            # SQL query using RAG text-to-SQL
            if self.sql_tool is None:
                observation = 'SQL tool is not configured. Cannot execute SQL queries.'
            else:
                # Reset SQL tool and disable cache to ensure fresh results
                self.sql_tool.reset()
                observation = self.sql_tool.query(query=argument, use_cache=False)
                log_head = f':violet[SQL Query] :red[{argument}]:violet[...]\n- '
        elif action_type.lower() == 'lookup':
            if self.json_mode:
                title, term = argument
                observation = self.retriever.lookup(title=title, term=term)
                log_head = f':violet[Lookup for] :red[{term}] :violet[in document] :red[{title}]:violet[...]\n- '
            else:
                try:
                    title, term = argument.split(',')
                    title = title.strip()
                    term = term.strip()
                    observation = self.retriever.lookup(title=title, term=term)
                    log_head = f':violet[Lookup for] :red[{term}] :violet[in document] :red[{title}]:violet[...]\n- '
                except Exception:
                    observation = f'Invalid argument format: {argument}. Must be in the format "title, term".'
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

    def forward(self, requirements: str, *args, **kwargs) -> str:
        while not self.is_finished():
            command = self._prompt_searcher(requirements=requirements)
            self.command(command)
        if not self.finished:
            return 'Searcher did not return any result.'
        return f'Search result: {self.results}'

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if not isinstance(argument, str):
            return f'Invalid argument type: {type(argument)}. Must be a string.'
        
        # Reset SQL tool to clear any cached state or history before processing new query
        # This ensures each query is processed independently without influence from previous prompts
        if self.sql_tool is not None:
            self.sql_tool.reset()
        
        # If query is database-related and SQL tool is available, query directly
        if self.sql_tool is not None:
            if self._is_sql_query(argument) or self._is_database_query(argument):
                logger.debug(f'Direct SQL query for database-related request: {argument}')
                # Pass use_cache=False to ensure fresh results for each query
                result = self.sql_tool.query(query=argument, use_cache=False)
                # Format result for Manager
                return f'Database query result: {result}'
        
        # Otherwise, use ReAct loop for general searches
        return self(requirements=argument)

if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    searcher = Searcher(config_path='config/agents/searcher.json', prompts=read_prompts('config/prompts/agent_prompt/react_search.json'))
    while True:
        requirements = input('Requirements: ')
        print(searcher(requirements=requirements))
