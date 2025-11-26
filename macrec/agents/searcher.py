from typing import Any, Optional
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.base import ToolAgent
from macrec.tools import Wikipedia, RAGTextToSQL, CollaborativeFiltering
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
    def cf_tool(self) -> Optional[CollaborativeFiltering]:
        """Optional Collaborative Filtering tool"""
        return self.tools.get('cf_tool', None)

    def _filter_candidates(self, argument: str) -> str:
        """Filter candidate items based on user preferences using advanced methods"""
        import re

        # Extract user_id from argument
        user_match = re.search(r'user (\d+)', argument)
        if not user_match:
            return "Could not extract user_id from filter request"

        user_id = int(user_match.group(1))

        # Determine filtering method
        method = 'hybrid'  # default
        if 'content-based' in argument.lower():
            method = 'content_based'
        elif 'popularity' in argument.lower():
            method = 'popularity'
        elif 'ml-based' in argument.lower():
            method = 'ml_based'
        elif 'genre' in argument.lower() or 'genres' in argument.lower():
            method = 'genres'

        # Extract user history if available (from recent interactions)
        user_history = []
        if self.sql_tool is not None:
            try:
                # Get user's recent interactions
                history_query = f"SELECT item_id FROM interactions WHERE user_id = {user_id} ORDER BY timestamp DESC LIMIT 20"
                history_result = self.sql_tool.query(query=history_query, use_cache=False)
                if isinstance(history_result, str):
                    history_ids = re.findall(r'\b\d{1,4}\b', history_result)
                    user_history = [int(x) for x in history_ids if 1 <= int(x) <= 2000][:10]
            except Exception as e:
                logger.warning(f"Could not get user history: {e}")

        # Extract preferred genres if specified
        preferred_genres = None
        genres_match = re.search(r'based on ([^)]+)', argument)
        if genres_match:
            genres_str = genres_match.group(1)
            preferred_genres = [g.strip() for g in genres_str.split(',') if g.strip()]

        # Apply filtering method
        try:
            if method == 'genres' and preferred_genres:
                # Use InfoDatabase for genre filtering
                from macrec.tools.info_database import InfoDatabase
                info_db = InfoDatabase()
                candidates = info_db.filter_candidates_by_genres(preferred_genres, limit=50)
                return f"Genre-filtered candidates for user {user_id}: {candidates[:20]}"

            elif method == 'content_based':
                from macrec.tools.info_database import InfoDatabase
                info_db = InfoDatabase()
                candidates = info_db.filter_candidates_content_based(user_history, limit=50)
                return f"Content-based filtered candidates for user {user_id}: {candidates[:20]}"

            elif method == 'popularity':
                from macrec.tools.info_database import InfoDatabase
                info_db = InfoDatabase()
                candidates = info_db.filter_candidates_by_popularity(limit=50)
                return f"Popularity-filtered candidates for user {user_id}: {candidates[:20]}"

            elif method == 'ml_based':
                from macrec.tools.info_database import InfoDatabase
                info_db = InfoDatabase()
                candidates = info_db.filter_candidates_ml_based(user_id, user_history, limit=50)
                return f"ML-based filtered candidates for user {user_id}: {candidates[:20]}"

            else:  # hybrid
                from macrec.tools.info_database import InfoDatabase
                info_db = InfoDatabase()
                candidates = info_db.filter_candidates_hybrid(user_id, user_history, preferred_genres, limit=50)
                return f"Hybrid-filtered candidates for user {user_id}: {candidates[:20]}"

        except Exception as e:
            logger.error(f"Advanced filtering failed: {e}")
            # Fallback to simple SQL-based filtering
            if self.sql_tool is not None and preferred_genres:
                genre_conditions = " OR ".join([f"Genres LIKE '%{genre}%'" for genre in preferred_genres])
                sql_query = f"SELECT item_id FROM movies WHERE {genre_conditions} LIMIT 50"

                try:
                    result = self.sql_tool.query(query=sql_query, use_cache=False)
                    if isinstance(result, str) and 'item_id' in result:
                        item_ids = re.findall(r'\b\d{1,4}\b', result)
                        item_ids = [int(x) for x in item_ids if 1 <= int(x) <= 2000][:20]
                        return f"Fallback SQL-filtered candidates for user {user_id}: {item_ids}"
                except Exception as e2:
                    logger.error(f"Fallback SQL filtering also failed: {e2}")

            return f"Error filtering candidates for user {user_id}: {str(e)}"

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
            # Check if this is a candidate filtering request
            if 'filter candidates' in argument.lower() and 'user' in argument.lower():
                # Extract user_id and preferences from argument
                observation = self._filter_candidates(argument)
                log_head = f':violet[Filter candidates] :red[{argument}]:violet[...]\n- '
            # Prioritize SQL tool if query is database-related
            elif self.sql_tool is not None:
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
        elif action_type.lower() == 'recommend' or action_type.lower() == 'cf':
            # CF Retrieval
            if self.cf_tool is None:
                observation = 'Collaborative Filtering tool is not configured. Cannot execute CF retrieval.'
            else:
                # Argument should be user_id
                try:
                    user_id = int(argument)
                    # Get top-k recommendations
                    recommendations = self.cf_tool.recommend_items_user_based(user_id=user_id, n_items=50) # Get more candidates for ranking
                    observation = f"CF Recommendations for User {user_id}: {recommendations}"
                    log_head = f':violet[CF Recommendation for User] :red[{user_id}]:violet[...]\n- '
                except ValueError:
                    observation = f"Invalid user_id for CF: {argument}. Must be an integer."
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
            if self._is_sql_query(argument) or self._is_database_query(argument) or self._is_personalized_query(argument):
                logger.debug(f'Direct SQL query for database-related request: {argument}')

                # Handle personalized queries with user context
                if self._is_personalized_query(argument):
                    query = self._build_personalized_query(argument)
                else:
                    query = argument

                # Pass use_cache=False to ensure fresh results for each query
                result = self.sql_tool.query(query=query, use_cache=False)

                # Try to extract item IDs from the result for CF optimization
                item_ids = self._extract_item_ids_from_sql_result(result)
                if item_ids:
                    logger.debug(f'Extracted {len(item_ids)} item IDs from search result: {item_ids[:10]}...')
                    # Store item IDs for later use by collaboration system
                    # Format result for Manager
                    return f'Database query result: {result}\n\nITEM_IDS: {",".join(map(str, item_ids))}'
                else:
                    return f'Database query result: {result}'

        # Otherwise, use ReAct loop for general searches
        return self(requirements=argument)

    def _is_personalized_query(self, argument) -> bool:
        """Check if query contains user personalization context."""
        if isinstance(argument, dict):
            return argument.get('personalized', False) or 'user_id' in argument

        if isinstance(argument, str):
            return 'user' in argument.lower() and ('based on their' in argument.lower() or 'preferences' in argument.lower())

        return False

    def _build_personalized_query(self, argument) -> str:
        """Build personalized query based on user context."""
        try:
            if isinstance(argument, dict):
                user_id = argument.get('user_id')
                base_query = argument.get('query', 'find movies')
            else:
                # Extract user_id from string like "filter candidates for user 2 based on Drama..."
                import re
                user_match = re.search(r'user (\d+)', str(argument), re.IGNORECASE)
                user_id = int(user_match.group(1)) if user_match else None
                base_query = str(argument)

            if user_id:
                # Create query that finds HIGHLY personalized candidates (20-50 items)
                personalized_query = f"""
                Find the TOP 30 most relevant movies for user {user_id} based on their preferences.

                STEPS:
                1. Analyze user {user_id}'s rating history - identify their favorite genres and highly rated movies (rating >= 4.0)
                2. Find movies that are MOST SIMILAR to their top-rated movies in terms of genre, style, and themes
                3. Prioritize movies they haven't rated yet
                4. Focus on quality over quantity - select only the 30 best matches
                5. Balance between their preferred genres (Drama, Comedy, Action) and some variety

                CRITERIA for top candidates:
                - High similarity to user's favorites
                - Unrated by the user
                - Good overall ratings/popularity
                - Diverse but relevant recommendations

                Return exactly 30 personalized movie recommendations that this user is most likely to enjoy.
                """
                logger.info(f"Created high-quality personalized query for user {user_id} (30 candidates)")
                return personalized_query
            else:
                return str(argument)
        except Exception as e:
            logger.warning(f"Failed to build personalized query: {e}")
            return str(argument)

    def _extract_item_ids_from_sql_result(self, result: str) -> list[int]:
        """
        Extract item IDs from SQL result for CF optimization
        """
        import re

        # Look for ITEM_IDS: 1,2,3,4 format first
        item_ids_match = re.search(r'ITEM_IDS:\s*([0-9,]+)', result)
        if item_ids_match:
            ids_str = item_ids_match.group(1)
            try:
                return [int(x.strip()) for x in ids_str.split(',') if x.strip()]
            except:
                pass

        # Fallback: extract from text patterns
        item_patterns = [
            r'item (\d+)',
            r'Item (\d+)',
            r'movie (\d+)',
            r'Movie (\d+)',
        ]

        candidates = []
        for pattern in item_patterns:
            matches = re.findall(pattern, result, re.IGNORECASE)
            candidates.extend([int(match) for match in matches])

        return list(set(candidates))[:50]  # Max 50 candidates

if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    searcher = Searcher(config_path='config/agents/searcher.json', prompts=read_prompts('config/prompts/agent_prompt/react_search.json'))
    while True:
        requirements = input('Requirements: ')
        print(searcher(requirements=requirements))
