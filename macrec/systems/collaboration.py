import json
import re
from typing import Any, Optional
from loguru import logger

from macrec.systems.base import System
from macrec.agents import Agent, Manager, Analyst, Interpreter, Reflector, Searcher, CandidateGenerator, RankingAgent, CollaborativeFilteringAgent, SequentialRecommendationAgent, SynthesizerAgent
from macrec.utils import parse_answer, parse_action, format_chat_history

class CollaborationSystem(System):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'gen', 'chat']

    def init(self, *args, **kwargs) -> None:
        """
        Initialize the ReAct system.
        """
        self.max_step: int = self.config.get('max_step', 10)
        self.require_search_before_finish: bool = self.config.get('require_search_before_finish', True)
        self.flow_type: str = self.config.get('flow_type', 'react')
        assert 'agents' in self.config, 'Agents are required.'
        self.init_agents(self.config['agents'])
        self.manager_kwargs = {
            'max_step': self.max_step,
            'reflections': '',  # Always initialize reflections
        }
        if self.interpreter is not None:
            self.manager_kwargs['task_prompt'] = ''

    def init_agents(self, agents: dict[str, dict]) -> None:
        self.agents: dict[str, Agent] = dict()
        for agent, agent_config in agents.items():
            try:
                agent_class = globals()[agent]
                assert issubclass(agent_class, Agent), f'Agent {agent} is not a subclass of Agent.'
                self.agents[agent] = agent_class(**agent_config, **self.agent_kwargs)
            except KeyError:
                raise ValueError(f'Agent {agent} is not supported.')
        assert 'Manager' in self.agents, 'Manager is required.'

    @property
    def manager(self) -> Optional[Manager]:
        if 'Manager' not in self.agents:
            return None
        return self.agents['Manager']

    @property
    def analyst(self) -> Optional[Analyst]:
        if 'Analyst' not in self.agents:
            return None
        return self.agents['Analyst']

    @property
    def interpreter(self) -> Optional[Interpreter]:
        if 'Interpreter' not in self.agents:
            return None
        return self.agents['Interpreter']

    @property
    def reflector(self) -> Optional[Reflector]:
        if 'Reflector' not in self.agents:
            return None
        return self.agents['Reflector']

    @property
    def searcher(self) -> Optional[Searcher]:
        if 'Searcher' not in self.agents:
            return None
        return self.agents['Searcher']

    @property
    def candidate_generator(self) -> Optional[CandidateGenerator]:
        if 'CandidateGenerator' not in self.agents:
            return None
        return self.agents['CandidateGenerator']
    
    @property
    def ranking_agent(self) -> Optional[RankingAgent]:
        if 'RankingAgent' not in self.agents:
            return None
        return self.agents['RankingAgent']

    @property
    def collaborative_filtering_agent(self) -> Optional[CollaborativeFilteringAgent]:
        if 'CollaborativeFilteringAgent' not in self.agents:
            return None
        return self.agents['CollaborativeFilteringAgent']

    @property
    def sequential_recommendation_agent(self) -> Optional[SequentialRecommendationAgent]:
        if 'SequentialRecommendationAgent' not in self.agents:
            return None
        return self.agents['SequentialRecommendationAgent']

    @property
    def synthesizer_agent(self) -> Optional[SynthesizerAgent]:
        if 'SynthesizerAgent' not in self.agents:
            return None
        return self.agents['SynthesizerAgent']

    def _detect_database_task(self, text: str) -> bool:
        """Detect if task is related to database queries"""
        if not text:
            return False
        text_lower = text.lower()
        
        # EXCLUSION: If the text is asking for clarification or stating input is meaningless, it's NOT a database task
        clarification_keywords = [
            'không có nghĩa', 'meaningless', 'rõ ràng hơn', 'clearer', 
            'cụ thể hơn', 'specific', 'nhập lại', 're-enter',
            'không hiểu', 'don\'t understand', 'xin chào', 'hello', 'hi'
        ]
        if any(k in text_lower for k in clarification_keywords):
            return False
            
        db_keywords = [
            # Explicit query intent
            'database', 'bảng', 'table', 'truy vấn', 'query', 'sql',
            'select', 'top', 'liệt kê', 'tìm', 'find', 'search', 'kiếm',
            'danh sách', 'list', 'thống kê', 'count',
            'tổng hợp', 'summary', 'lọc', 'filter', 'sắp xếp', 'sort',
            'gợi ý', 'suggest', 'recommend', 'tiếp theo', 'next', 'tương tự', 'similar',
            
            # Domain specific keywords (Movies/Recommendation)
            'phim', 'movie', 'film', 'cinema',
            'đạo diễn', 'director', 'diễn viên', 'actor', 'actress', 'cast',
            'thể loại', 'genre', 'category',
            'đánh giá', 'rating', 'review', 'score',
            'người dùng', 'user', 'khách hàng', 'customer',
            'sản phẩm', 'item', 'product'
        ]
        return any(keyword in text_lower for keyword in db_keywords)
    
    def _is_generic_recommendation(self, text: str) -> bool:
        """Detect if this is a generic recommendation without user context"""
        if not text:
            return False
        text_lower = text.lower()
        
        # Check for user-specific keywords
        has_user_context = any(k in text_lower for k in [
            'user_', 'user ', 'i like', 'i watched', 'i just', 'i have',
            'my ', 'me ', 'tôi ', 'của tôi', 'tôi thích', 'tôi vừa'
        ])
        
        # Check for generic recommendation keywords
        has_generic_keywords = any(k in text_lower for k in [
            'best', 'top', 'recommend', 'gợi ý', 'hay nhất', 'tốt nhất',
            'suggest', 'popular', 'phổ biến'
        ])
        
        return has_generic_keywords and not has_user_context
    
    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.step_n: int = 1
        self._search_performed: bool = False
        self._analyse_performed: bool = False
        self._sequential_performed: bool = False
        self._is_database_task: bool = False
        self._candidates: list[int] = []
        self._current_user_id: Optional[int] = None
        if clear:
            if self.reflector is not None:
                self.reflector.reflections = []
                self.reflector.reflections_str = ''
            if self.task == 'chat':
                self._chat_history = []
        
        if self.task == 'chat':
            # Set a default answer in case the system halts immediately
            self.answer = "I apologize, but I couldn't process your request."

    def add_chat_history(self, chat: str, role: str) -> None:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        self._chat_history.append((chat, role))

    @property
    def chat_history(self) -> list[tuple[str, str]]:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        return format_chat_history(self._chat_history)

    def is_halted(self) -> bool:
        over_limit = self.manager.over_limit(scratchpad=self.scratchpad, **self.manager_kwargs)
        print(f"DEBUG: is_halted check - Step: {self.step_n}, Max Step: {self.max_step}, Over Limit: {over_limit}, Finished: {self.finished}")
        return ((self.step_n > self.max_step) or over_limit) and not self.finished

    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer if self.task != 'chat' else '', json_mode=self.manager.json_mode, **self.kwargs)

    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)

    def act(self) -> tuple[str, Any]:
        # Act
        # Check for empty input at step 1
        input_text = self.manager_kwargs.get('input', '')
        if self.step_n == 1 and input_text is not None and not input_text.strip():
            logger.warning("Empty input detected in act(), returning Finish action.")
            return 'finish', "Please provide a valid input."

        # Detect if this is a database-related task
        if not hasattr(self, '_is_database_task') or self.step_n == 1:
            task_prompt = self.manager_kwargs.get('task_prompt', '')
            self._is_database_task = self._detect_database_task(task_prompt + ' ' + input_text)
        
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        
        # Strong reminder for database tasks
        if self._is_database_task and not (self._search_performed or self._analyse_performed or getattr(self, '_sequential_performed', False)):
            self.scratchpad += '\n⚠️ CRITICAL: This task requires querying the database or analyzing data. You MUST use Search[...] or Analyse[...] or Sequential[...] action first. You CANNOT Finish without searching or analyzing first.'
            
            # Specific hint for sequential recommendation
            task_prompt_lower = (self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')).lower()
            sequential_keywords = ['tiếp theo', 'next', 'sequential', 'what next', 'what to watch next', 'next items']
            
            if any(k in task_prompt_lower for k in sequential_keywords):
                # Extract user_id if present
                import re
                user_match = re.search(r'user[_\s]?(\d+)', task_prompt_lower)
                if user_match:
                    user_id = user_match.group(1)
                    self.scratchpad += f'\n💡 HINT: This is a sequential recommendation query. You should call Sequential[{user_id}, K] (replace K with desired number) immediately.'
                else:
                    self.scratchpad += '\n💡 HINT: This is a sequential recommendation query. Call Sequential[user_id, K] immediately.'
            elif any(k in task_prompt_lower for k in ['gợi ý', 'suggest', 'recommend', 'tương tự', 'similar']):
                 self.scratchpad += '\n💡 HINT: If the user mentions a specific item (movie, product), you should first Search for its ID (e.g., Search[What is the item_id of movie X?]), then use that ID to Analyse or Search for similar items.'
        
        # NEW: Detect user_id in query and enforce Analyse, but skip for sequential recommendation queries
        import re
        combined_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
        user_id_match = re.search(r'user[_\s]?(\d+)', combined_text, re.IGNORECASE)
        
        # Determine if this is a sequential recommendation request
        sequential_keywords = ['tiếp theo', 'next', 'sequential', 'what next', 'what to watch next', 'next items']
        is_sequential = any(k in combined_text.lower() for k in sequential_keywords)
        
        if user_id_match and not self._analyse_performed and not getattr(self, '_sequential_performed', False) and not is_sequential:
            user_id = user_id_match.group(1)
            self.scratchpad += f'\n🚨 MANDATORY: Query mentions user_{user_id}. You MUST call Analyse[user, {user_id}] FIRST to get their profile and history before making any recommendations. This is NOT optional.'
        
        # Check if previous thought mentioned Search but haven't searched yet
        if self.require_search_before_finish and not (self._search_performed or self._analyse_performed or getattr(self, '_sequential_performed', False)):
            last_thought = self.scratchpad.split('Thought')[-1].split('Action')[0].lower() if 'Thought' in self.scratchpad else ''
            if 'search' in last_thought and 'finish' not in last_thought:
                # Thought mentioned search, remind to use Search action
                self.scratchpad += '\nREMINDER: Your Thought mentioned searching. You MUST use Search[...] action now, not Finish.'
        
        # NEW: Early hint for generic recommendations (before search)
        task_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
        if self._is_generic_recommendation(task_text) and not self._search_performed:
            self.scratchpad += '\n🎯 GENERIC RECOMMENDATION DETECTED: This query asks for general "best" recommendations WITHOUT user context. You MUST use Search[query] to find candidates, then Finish directly with results. DO NOT call Analyse or CF.'
        
        # NEW: Strong hint for generic recommendations with search results
        if (self._search_performed and 
            hasattr(self, '_candidates') and 
            len(self._candidates) >= 5):
            
            if self._is_generic_recommendation(task_text):
                self.scratchpad += f'\n🚨 MANDATORY: You have {len(self._candidates)} candidates from Search for a GENERIC recommendation. You MUST Finish NOW with these results. DO NOT call Analyse, CF, or any other agent. Calling additional agents wastes resources and is INCORRECT for generic queries.'
        
        # Check if we should add efficiency hint for generic recommendations
        task_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
        is_generic = self._is_generic_recommendation(task_text)
        
        # Early hint for generic recommendations (before search)
        if is_generic and not self._search_performed:
            hint = "\n\nEFFICIENCY HINT: This is a GENERIC recommendation query (no specific user_id). You should:\n1. Search for the best items matching the query\n2. Finish IMMEDIATELY with the top results from Search\nDO NOT call Analyse[user] or CF for generic queries."
            self.scratchpad += hint
        
        # Mandatory hint after search for generic recommendations
        if is_generic and self._search_performed and self._candidates:
            hint = f"\n\nMANDATORY: You have search results with {len(self._candidates)} candidates. This is a GENERIC query. You MUST call Finish NOW with the top 5 items from search results. DO NOT call Analyse or CF."
            self.scratchpad += hint
        
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        
        # AUTO-FINISH: If we have CF results and Manager didn't Finish, auto-finish
        if hasattr(self, '_cf_results') and self._cf_results and self.step_n >= self.max_step - 1:
            if 'Finish[' not in action:
                # Manager failed to Finish, auto-finish with CF results
                recs = self._cf_results['detailed']
                top_5 = recs[:5]
                auto_response = f"Here are 5 movie recommendations for user {self._cf_results['user_id']}:\n" + "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(top_5)])
                
                # Add Summaries
                auto_response += self._summarize_recommendations(auto_response)
                
                # Add Genre Analysis
                auto_response += self._analyze_user_genres(self._cf_results['user_id'])
                
                logger.info("AUTO-FINISH: Manager didn't Finish, using CF results")
                action = f"Finish[{auto_response}]"
        
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        
        # Auto-convert Finish to Search for database tasks
        if action_type.lower() == 'finish' and self._is_database_task and not (self._search_performed or self._analyse_performed or getattr(self, '_sequential_performed', False)):
            logger.warning(f'Auto-converting Finish to Search for database task. Original: {action}')
            # Generate a suggested SQL query based on task
            task_prompt = self.manager_kwargs.get('task_prompt', '')
            suggested_query = self._generate_sql_suggestion(task_prompt + ' ' + input_text)
            action_type = 'search'
            argument = suggested_query
            action = f'Search[{suggested_query}]'
            self.scratchpad += '\n⚠️ AUTO-CORRECTED: Finish action was converted to Search action because this is a database query task.'
        
        logger.debug(f'Action {self.step_n}: {action}')
        
        # DEBUG LOGS
        print(f"DEBUG: Step {self.step_n}")
        print(f"DEBUG: Action: {action}")
        print(f"DEBUG: Is Database Task: {self._is_database_task}")
        print(f"DEBUG: Search Performed: {self._search_performed}")
        print(f"DEBUG: Analyse Performed: {self._analyse_performed}")

        return action_type, argument
    
    def _generate_sql_suggestion(self, task_text: str) -> str:
        """Generate a suggested natural language query based on task text"""
        # Return the task text as-is, as it's already in natural language
        # The RAG SQL tool will handle SQL generation
        return task_text

    def execute(self, action_type: str, argument: Any):
        # Execute
        log_head = ''
        if action_type.lower() == 'finish':
            # UPDATED CHECK: For database tasks, we need EITHER search OR CF results
            has_cf_results = hasattr(self, '_cf_results') and self._cf_results is not None
            
            if self._is_database_task and not self._search_performed and not has_cf_results and not getattr(self, '_sequential_performed', False):
                 # Provide helpful guidance when Finish is blocked
                task_prompt = self.manager_kwargs.get('task_prompt', '')
                input_text = self.manager_kwargs.get('input', '')
                combined_text = (task_prompt + ' ' + input_text).lower()
                
                # Check if this is a recommendation task
                is_recommendation = any(keyword in combined_text for keyword in [
                    'recommend', 'suggest', 'gợi ý', 'đề xuất', 'nên xem'
                ])
                
                if is_recommendation:
                    observation = (
                        'Finish action blocked: You MUST use Search[...] or CF[...] to find actual items from the database before finishing. '
                        'Analysis alone is not enough. For personalized recommendations, use: Analyse[user] → CF[user_id, top_k] → Finish.'
                    )
                else:
                    observation = (
                        'Finish action blocked: You MUST use Search[...] to find actual items from the database before finishing. '
                        'Analysis alone is not enough.'
                    )
                log_head = ':violet[Finish blocked - must Search or CF first]:\n- '
            elif self.require_search_before_finish and self._is_database_task and not (self._search_performed or self._analyse_performed or getattr(self, '_sequential_performed', False)):
                # Legacy check - now only applied if we detected it IS a database task but missed the first check for some reason
                # or if we want to be extra safe. 
                # Ideally, for non-database tasks (chit-chat), we should ALLOW Finish.
                observation = 'Finish action blocked: you must call Search[...] or Analyse[...] at least once before finishing.'
                log_head = ':violet[Finish blocked - must Search/Analyse first]:\n- '
            else:
                # NEW: Smart Finish for generic recommendations
                # If we have search results and this is a generic recommendation, use candidates directly
                if (self._search_performed and 
                    hasattr(self, '_candidates') and 
                    len(self._candidates) >= 5 and
                    not self._current_user_id):
                    
                    task_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
                    if self._is_generic_recommendation(task_text):
                        # Check if argument is empty or doesn't contain valid item IDs
                        if isinstance(argument, str):
                            # Try to extract item IDs from argument
                            import re
                            arg_ids = re.findall(r'\d+', argument)
                            
                            # If argument has no IDs or very few, use our candidates
                            if len(arg_ids) < 5:
                                # Use top 5 from candidates
                                top_5_ids = self._candidates[:5]
                                argument = ','.join(map(str, top_5_ids))
                                logger.info(f"Smart Finish: Using search candidates for generic recommendation: {argument}")
                                self.scratchpad += f'\n✅ AUTO-ENHANCED: Using top 5 items from Search results: {argument}'
                                
                                # Add Summaries (Generic)
                                # Get titles first
                                item_texts = []
                                for item_id in top_5_ids:
                                    if self.analyst and hasattr(self.analyst, 'info_retriever'):
                                        item_texts.append(self.analyst.info_retriever.item_info(item_id=item_id))
                                summaries = self._summarize_recommendations("\n".join(item_texts))
                                argument += summaries
                
                # DISABLED: Smart Finish logic conflicts with Manager's strict 4-part format
                # Manager now handles all recommendation formatting
                
                parse_result = self._parse_answer(argument)
                if parse_result['valid']:
                    enhanced_answer = parse_result['answer']

                    # Ensure Manager response includes user preference summary based on Analyst findings
                    if self._should_add_user_summary(enhanced_answer):
                        analyst_summary = self._extract_analyst_findings()
                        if analyst_summary:
                            enhanced_answer = self._add_user_preference_summary(enhanced_answer, analyst_summary)

                    observation = self.finish(enhanced_answer)
                    log_head = ':violet[Finish with enhanced answer]:\n- '
                else:
                    assert "message" in parse_result, "Invalid parse result."
                    observation = f'{parse_result["message"]} Valid Action examples are {self.manager.valid_action_example}.'
        elif action_type.lower() == 'analyse':
            if self.analyst is None:
                observation = 'Analyst is not configured. Cannot execute the action "Analyse".'
            else:
                # NEW: Block Analyst[user] for generic recommendations
                task_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
                is_generic = self._is_generic_recommendation(task_text)
                
                # Check if trying to analyze user without user_id
                is_user_analysis = False
                if self.manager.json_mode:
                    if isinstance(argument, list) and len(argument) >= 1:
                        is_user_analysis = argument[0].lower() == 'user'
                else:
                    if isinstance(argument, str):
                        parts = argument.split(',')
                        is_user_analysis = parts[0].strip().lower() == 'user'
                
                if is_generic and is_user_analysis:
                    observation = 'Analyst[user] blocked: This is a GENERIC recommendation without user context. You cannot analyze a user without a specific user_id. For generic "best" recommendations, use Search results directly and call Finish.'
                    log_head = ':violet[Analyst blocked - generic query]:\n- '
                else:
                    # Extract user_id from argument for personalization
                    if self.manager.json_mode:
                        if isinstance(argument, list) and len(argument) >= 2:
                            if argument[0].lower() == 'user':
                                self._current_user_id = argument[1] if isinstance(argument[1], int) else None
                    else:
                        if isinstance(argument, str):
                            parts = argument.split(',')
                            if len(parts) >= 2 and parts[0].strip().lower() == 'user':
                                try:
                                    self._current_user_id = int(parts[1].strip())
                                except:
                                    self._current_user_id = None

                    self.log(f':violet[Calling] :red[Analyst] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                    observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
                    log_head = f':violet[Response from] :red[Analyst] :violet[with] :blue[{argument}]:violet[:]\n- '
                    self._analyse_performed = True
        elif action_type.lower() == 'search':
            if self.searcher is None:
                observation = 'Searcher is not configured. Cannot execute the action "Search".'
            else:
                # Enhance search argument with user context for personalized results
                enhanced_argument = argument
                if hasattr(self, '_current_user_id') and self._current_user_id is not None:
                    if self.manager.json_mode:
                        # For JSON mode, pass user context
                        if isinstance(argument, str):
                            enhanced_argument = {
                                "query": argument,
                                "user_id": self._current_user_id,
                                "personalized": True
                            }
                    else:
                        # For text mode, append user context
                        enhanced_argument = f"{argument} for user {self._current_user_id} based on their preferences"

                self.log(f':violet[Calling] :red[Searcher] :violet[with] :blue[{enhanced_argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.searcher.invoke(argument=enhanced_argument, json_mode=self.manager.json_mode)
                # Try to extract candidates from search result
                self._extract_candidates_from_search(observation)

                # Log candidate quality info
                if self._candidates:
                    logger.info(f"Generated {len(self._candidates)} personalized candidates for recommendation")
                    if len(self._candidates) > 50:
                        logger.info("Note: Will use top 50 candidates for CF optimization")

                log_head = f':violet[Response from] :red[Searcher] :violet[with] :blue[{argument}]:violet[:]\n- '
                self._search_performed = True
        elif action_type.lower() == 'interpret':
            if self.interpreter is None:
                observation = 'Interpreter is not configured. Cannot execute the action "Interpret".'
            else:
                self.log(f':violet[Calling] :red[Interpreter] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.interpreter.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[Interpreter] :violet[with] :blue[{argument}]:violet[:]\n- '
        elif action_type.lower() == 'generatecandidates' or action_type.lower() == 'candidates':
            if self.candidate_generator is None:
                observation = 'CandidateGenerator is not configured. Cannot execute the action "GenerateCandidates".'
            else:
                self.log(f':violet[Calling] :red[CandidateGenerator] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.candidate_generator.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[CandidateGenerator] :violet[with] :blue[{argument}]:violet[:]\n- '
        elif action_type.lower() == 'rank' or action_type.lower() == 'ranking':
            if self.ranking_agent is None:
                observation = 'RankingAgent is not configured. Cannot execute the action "Rank".'
            else:
                self.log(f':violet[Calling] :red[RankingAgent] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.ranking_agent.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[RankingAgent] :violet[with] :blue[{argument}]:violet[:]\n- '
        elif action_type.lower() == 'cf' or action_type.lower() == 'collaborativefiltering':
            if self.collaborative_filtering_agent is None:
                observation = 'CollaborativeFilteringAgent is not configured. Cannot execute the action "CF".'
            else:
                # NEW: Block CF for generic recommendations
                task_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
                is_generic = self._is_generic_recommendation(task_text)
                
                if is_generic:
                    observation = 'CF action blocked: This is a GENERIC recommendation without user context. Collaborative Filtering requires a specific user_id. For generic "best" recommendations, use Search results directly and call Finish with the top items.'
                    log_head = ':violet[CF blocked - generic query]:\n- '
                else:
                    # BYPASS CF Agent LLM - Call CF Tool directly
                    try:
                        # Parse argument
                        if self.manager.json_mode:
                            if isinstance(argument, dict):
                                user_id = argument.get('user_id')
                                top_k = argument.get('top_k', 5)
                            else:
                                parts = str(argument).split(',')
                                user_id = int(parts[0])
                                top_k = int(parts[1]) if len(parts) > 1 else 5
                        else:
                            parts = argument.split(',')
                            user_id = int(parts[0])
                            top_k = int(parts[1]) if len(parts) > 1 else 5
                        
                        # Get candidates from previous search (if available)
                        candidates = self._candidates if self._candidates else None
                        
                        # Call CF Tool directly
                        cf_tool = self.collaborative_filtering_agent.cf_tool
                        recommendations = cf_tool.recommend_items_user_based(
                            user_id=user_id,
                            n_items=top_k,
                            method='pearson',
                            candidates=candidates  # Use search candidates if available
                        )
                        
                        # Format output
                        if recommendations:
                            detailed_recs = []
                            for item_id, score in recommendations[:top_k]:
                                item_info = self.collaborative_filtering_agent.info_retriever.item_info(item_id=item_id)
                                detailed_recs.append(f"{item_id}: {item_info} (score: {score:.2f})")
                            observation = f"CF Recommendations for User {user_id} (top {top_k}):\n" + "\n".join(detailed_recs)
                            # Note: We don't add summary/analysis here yet, we wait for Finish action (or Auto-Finish)
                            
                            # Store CF results for potential auto-finish
                            self._cf_results = {
                                'user_id': user_id,
                                'recommendations': recommendations,
                                'detailed': detailed_recs
                            }
                        else:
                            observation = f"No CF recommendations available for user {user_id}"
                            self._cf_results = None
                        
                        log_head = f':violet[CF Recommendation for User] :red[{user_id}]:violet[...]\n- '
                        self._analyse_performed = True
                        
                    except Exception as e:
                        observation = f"CF error: {str(e)}"
                        log_head = ':violet[CF Error]:\n- '
                        self._cf_results = None
        elif action_type.lower() == 'sequential':
            if self.sequential_recommendation_agent is None:
                observation = 'SequentialRecommendationAgent is not configured. Cannot execute the action "Sequential".'
            else:
                self.log(f':violet[Calling] :red[SequentialRecommendationAgent] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.sequential_recommendation_agent.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[SequentialRecommendationAgent] :violet[with] :blue[{argument}]:violet[:]\n- '
                self._sequential_performed = True
        elif action_type.lower() == 'synthesize':
            if self.synthesizer_agent is None:
                observation = 'SynthesizerAgent is not configured. Cannot execute the action "Synthesize".'
            else:
                self.log(f':violet[Calling] :red[SynthesizerAgent] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.synthesizer_agent.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[SynthesizerAgent] :violet[with] :blue[{argument}]:violet[:]\n- '
        else:
            logger.debug(f'Invalid action_type: "{action_type}", argument: "{argument}"')
            observation = f'Invalid Action type or format. Valid Action examples are {self.manager.valid_action_example}.'

        self.scratchpad += f'\nObservation: {observation}'

        logger.debug(f'Observation: {observation}')
        self.log(f'{log_head}{observation}', agent=self.manager, logging=False)

    def _extract_candidates_from_search(self, observation: str) -> None:
        """Extract candidate item IDs from search observation."""
        import re

        # Store full observation for later use in Finish action
        self._last_search_observation = observation

        # First, try to extract from ITEM_IDS: format
        item_ids_match = re.search(r'ITEM_IDS:\s*([0-9,]+)', observation)
        if item_ids_match:
            ids_str = item_ids_match.group(1)
            try:
                self._candidates = [int(x.strip()) for x in ids_str.split(',') if x.strip()]
                logger.debug(f'Extracted {len(self._candidates)} candidates from ITEM_IDS: {self._candidates[:10]}...')
                return
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
            matches = re.findall(pattern, observation, re.IGNORECASE)
            candidates.extend([int(match) for match in matches])

        # Remove duplicates and limit to reasonable number
        self._candidates = list(set(candidates))[:50]  # Max 50 candidates
        if self._candidates:
            logger.debug(f'Extracted {len(self._candidates)} candidates from text patterns: {self._candidates[:10]}...')

    def _is_candidates_in_argument(self, argument) -> bool:
        """Check if candidates are already included in the CF argument."""
        if self.manager.json_mode:
            return isinstance(argument, dict) and 'candidates' in argument
        else:
            return isinstance(argument, str) and '[' in argument and ']' in argument

    def step(self):
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        
        # Force finish if we are at max step and not finished
        if self.step_n >= self.max_step and not self.finished:
             self.finish("I apologize, I could not complete the task within the limit.")
             
        self.step_n += 1

    def reflect(self) -> bool:
        if (not self.is_finished() and not self.is_halted()) or self.reflector is None:
            self.reflected = False
            if self.reflector is not None:
                self.manager_kwargs['reflections'] = ''
            return False
        self.reflector(self.input, self.scratchpad)
        self.reflected = True
        self.manager_kwargs['reflections'] = self.reflector.reflections_str
        if self.reflector.json_mode:
            reflection_json = json.loads(self.reflector.reflections[-1])
            if 'correctness' in reflection_json and reflection_json['correctness']:
                # don't forward if the last reflection is correct
                logger.debug('Last reflection is correct, don\'t forward.')
                self.log(":red[**Last reflection is correct, don't forward**]", agent=self.reflector, logging=False)
                return True
        return False

    def interprete(self) -> None:
        if self.task == 'chat':
            assert self.interpreter is not None, 'Interpreter is required for chat task.'
            self.manager_kwargs['task_prompt'] = self.interpreter(input=self.chat_history)
        else:
            if self.interpreter is not None:
                self.manager_kwargs['task_prompt'] = self.interpreter(input=self.input)

    def sequential_forward(self, user_input: str) -> str:
        """
        Execute the sequential flow:
        User -> Interpreter -> Manager -> Searcher -> Analyst -> Reflector -> Interpreter -> User
        """
        # Only clear history if NOT chat task
        is_chat = self.task == 'chat'
        self.reset(clear=not is_chat)
        
        # Add user input to history if chat
        if is_chat:
            self.add_chat_history(user_input, role='user')
            
        self.input = user_input
        
        # 1. Interpreter (NLU)
        # Use Interpreter to understand the query and generate a task prompt
        if self.interpreter:
            if is_chat:
                # Use history for context in chat mode
                task_prompt = self.interpreter(input=self.chat_history)
            else:
                task_prompt = self.interpreter(input=user_input)
        else:
            task_prompt = user_input
            
        # 2. Manager (Router)
        # Determine intent based on task_prompt
        # Simple keyword-based routing for now, as Manager is a ReAct agent
        is_recommendation = self._detect_database_task(task_prompt)
        
        if is_recommendation:
            # 3. Searcher (Candidate Generation)
            if self.searcher:
                # We use invoke to get the result. Searcher handles Hybrid search (Vector/SQL/CF) internally if configured.
                search_result = self.searcher.invoke(argument=task_prompt, json_mode=False)
            else:
                search_result = "Searcher not configured."
                
            # 4. Analyst (Ranking)
            # Only run if we have a ranking agent and search result looks like a list of items
            # For now, we'll pass the search result to the Reflector/Interpreter directly
            # unless we can parse IDs. 
            # To support the diagram, we'll try to invoke RankingAgent if search_result is a list-like string.
            ranked_result = search_result
            if self.ranking_agent:
                 # Try to parse IDs from search_result if it's a string representation of a list
                 # This is a placeholder for actual ID extraction logic
                 # For now, we assume RankingAgent can handle the context or we skip explicit ranking call 
                 # if we can't parse candidates.
                 pass

            # 5. Reflector (Validation)
            if self.reflector:
                self.reflector(input=task_prompt, scratchpad=ranked_result)
                if self.reflector.reflections:
                     # If validation fails or has feedback, we could loop back.
                     # For this implementation, we just append reflection to context.
                     ranked_result += f"\nReflector Feedback: {self.reflector.reflections_str}"

            # 6. Interpreter (NLG)
            # Generate final response
            if self.interpreter:
                final_response = self.interpreter.invoke(argument=f"Query: {user_input}\nContext: {ranked_result}", json_mode=False)
            else:
                final_response = str(ranked_result)
        else:
            # Chat/Explain Flow
            if self.interpreter:
                final_response = self.interpreter.invoke(argument=user_input, json_mode=False)
            else:
                final_response = "Interpreter not configured for chat."
        
        # Add system response to history if chat
        if is_chat:
            self.add_chat_history(final_response, role='system')
            
        return final_response

    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        if self.flow_type == 'sequential' and user_input is not None:
             return self.sequential_forward(user_input)

        if self.task == 'chat':
            self.manager_kwargs['history'] = self.chat_history
            self.manager_kwargs['input'] = user_input
        else:
            self.manager_kwargs['input'] = self.input
        if self.reflect():
            return self.answer
        if reset:
            self.reset()
        if self.task == 'chat':
            assert user_input is not None, 'User input is required for chat task.'
            self.add_chat_history(user_input, role='user')
        self.interprete()
        while not self.is_finished() and not self.is_halted():
            self.step()
        if self.task == 'chat':
            # If answer is empty after execution, provide a fallback
            if not self.answer:
                self.answer = "I apologize, but I couldn't process your request. Please try rephrasing."
            self.add_chat_history(self.answer, role='system')
        return self.answer

    def chat(self) -> None:
        assert self.task == 'chat', 'Chat task is required for chat method.'
        print("Start chatting with the system. Type 'exit' or 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
    def _summarize_recommendations(self, recommendations: str) -> str:
        """
        Generate detailed descriptions for recommended movies using the Manager LLM.
        """
        if not recommendations or recommendations.strip() == "":
            return ""

        # Validate that we only describe movies that are actually in the recommendations
        valid_movies = []
        for line in recommendations.split('\n'):
            line = line.strip()
            if line and not line.startswith('Movie:'):
                # Extract movie title from patterns like "Title (Year)" or just "Title"
                title_match = re.search(r'^([^(]+)', line)
                if title_match:
                    title = title_match.group(1).strip()
                    # Skip generic titles that might be hallucinations
                    if title.lower() not in ['copycat', 'movie', 'film', 'the movie']:
                        valid_movies.append(line)

        if not valid_movies:
            return ""

        clean_recommendations = '\n'.join(valid_movies[:5])  # Limit to 5 movies max

        prompt = f"""ONLY describe the movies listed below. DO NOT describe any other movies or invent new ones.

For EACH movie in this EXACT list, provide a DETAILED description:

1. **Plot Summary**: 2-3 sentences describing the main story and key events
2. **Genre**: Main genre and any sub-genres
3. **Key Cast**: Main actors/actresses and director if mentioned
4. **Why Recommended**: Why this movie would appeal to the user
5. **Notable Info**: Awards, critical reception, cultural impact, or interesting facts

Format each movie as a separate paragraph with clear headings. Make descriptions engaging and informative. Use natural, conversational language.

MOVIES TO DESCRIBE (ONLY THESE - DO NOT ADD OTHERS):
{clean_recommendations}

IMPORTANT: Only describe movies from the list above. If a movie is not in the list, do NOT describe it."""

        # Use Manager's thought LLM directly to generate descriptions
        summary = self.manager.thought_llm(prompt)

        # Additional validation: remove any descriptions not in our valid list
        validated_summary = self._validate_movie_summaries(summary, valid_movies)

        return f"\n\n🎬 **Chi tiết phim gợi ý:**\n{validated_summary}"

    def _should_add_user_summary(self, response: str) -> bool:
        """Check if response needs user preference summary added."""
        if not response:
            return False

        # Check if response already has user preference summary
        if "**USER PREFERENCE SUMMARY**" in response or "USER PREFERENCE SUMMARY" in response:
            return False

        # Check if this is a movie recommendation response
        response_lower = response.lower()
        return ("recommend" in response_lower or "gợi ý" in response_lower) and "movie" in response_lower

    def _extract_analyst_findings(self) -> str:
        """Extract user preferences from the most recent Analyst response."""
        if not hasattr(self, 'scratchpad') or not self.scratchpad:
            return ""

        # Find the last Analyst response in scratchpad
        lines = self.scratchpad.split('\n')
        analyst_response = ""

        for i, line in enumerate(reversed(lines)):
            if "Response from Analyst" in line:
                # Extract the Analyst response (next few lines)
                start_idx = len(lines) - i - 1
                for j in range(start_idx + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith("👩‍💼Manager:"):
                        analyst_response += lines[j] + " "
                    elif lines[j].startswith("👩‍💼Manager:"):
                        break
                break

        if not analyst_response.strip():
            return ""

        # Extract key information from Analyst response
        analyst_lower = analyst_response.lower()

        # Try to find specific patterns in Analyst response
        preferences = []

        # Extract genres mentioned
        genres = []
        if "drama" in analyst_lower:
            genres.append("Drama")
        if "action" in analyst_lower:
            genres.append("Action")
        if "comedy" in analyst_lower:
            genres.append("Comedy")
        if "thriller" in analyst_lower:
            genres.append("Thriller")

        # Extract quality indicators
        quality_indicators = []
        if "high-quality" in analyst_lower or "quality" in analyst_lower:
            quality_indicators.append("high-quality content")
        if "engaging" in analyst_lower:
            quality_indicators.append("engaging narratives")
        if "emotional" in analyst_lower or "emotion" in analyst_lower:
            quality_indicators.append("emotionally resonant stories")
        if "character development" in analyst_lower or "character" in analyst_lower:
            quality_indicators.append("strong character development")

        # Build preference summary
        summary_parts = []

        if genres:
            summary_parts.append(f"films in {', '.join(genres)} genres")

        if quality_indicators:
            summary_parts.extend(quality_indicators)

        if summary_parts:
            preference_str = ", ".join(summary_parts)
            return f"Based on Analyst analysis, user prefers {preference_str}."
        else:
            # Fallback: try to extract meaningful sentences from Analyst response
            sentences = analyst_response.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and ('prefers' in sentence.lower() or 'likes' in sentence.lower() or 'enjoys' in sentence.lower()):
                    return f"Based on Analyst analysis: {sentence}"

            return "Based on Analyst analysis, user prefers high-quality films with engaging content."

    def _add_user_preference_summary(self, response: str, analyst_summary: str) -> str:
        """Add user preference summary to the beginning of response in proper format."""
        if not response or not analyst_summary:
            return response

        # If response already has proper structure, don't modify
        if "**USER PREFERENCE SUMMARY**" in response:
            return response

        # Add user preference summary as the first part
        # Format the entire response properly
        formatted_response = f"1. **USER PREFERENCE SUMMARY**: {analyst_summary}\n\n"

        # If response looks like movie recommendations, make it part 2
        if "recommendations" in response.lower() or "here are" in response.lower() or response.strip().startswith("1."):
            formatted_response += f"2. **5 MOVIE RECOMMENDATIONS**:\n{response}\n\n"
            formatted_response += "3. **SELECTION CRITERIA**: These 5 movies selected for highest ratings from 30 recommendations and matching user preferences identified by Analyst.\n\n"
            formatted_response += "4. **FRIENDLY CLOSING**: Enjoy your movie watching experience!"
        else:
            # For other types of responses, just add the summary
            formatted_response += response

        return formatted_response

    def _validate_movie_summaries(self, summary: str, valid_movies: list) -> str:
        """Validate that summary only contains descriptions for valid movies."""
        if not summary or not valid_movies:
            return ""

        # Extract valid movie titles
        valid_titles = set()
        for movie in valid_movies:
            title_match = re.search(r'^([^(]+)', movie.strip())
            if title_match:
                title = title_match.group(1).strip().lower()
                valid_titles.add(title)

        # Filter summary to only include valid movies
        lines = summary.split('\n')
        filtered_lines = []
        current_movie_valid = False

        for line in lines:
            line_lower = line.lower().strip()
            # Check if this line starts a new movie description
            if any(title in line_lower for title in valid_titles):
                current_movie_valid = True
                filtered_lines.append(line)
            elif line.strip() == "" or line.startswith('**') or line.startswith('*'):
                # Keep formatting and empty lines
                if current_movie_valid:
                    filtered_lines.append(line)
            elif current_movie_valid:
                filtered_lines.append(line)
            else:
                # Skip lines that don't belong to valid movies
                continue

        return '\n'.join(filtered_lines)

    def _contains_movie_recommendations(self, text: str) -> bool:
        """Check if the response contains movie recommendations."""
        if not text:
            return False

        text_lower = text.lower()
        movie_keywords = [
            'phim', 'movie', 'film', 'cinema',
            'gợi ý', 'recommend', 'suggest',
            'xem', 'watch', 'nên xem', 'hay'
        ]

        # Check for movie-related keywords
        has_movie_keywords = any(keyword in text_lower for keyword in movie_keywords)

        # Check for item IDs (common in recommendations)
        import re
        has_item_ids = bool(re.search(r'\b\d{1,4}\b', text))

        return has_movie_keywords or has_item_ids

    def _is_personalized_recommendation(self, text: str) -> bool:
        """Check if this is a personalized recommendation for a specific user."""
        if not text:
            return False

        # Check for user-specific indicators
        personalized_indicators = [
            'user', 'người dùng', 'dựa trên sở thích',
            'based on your preferences', 'for user',
            'theo sở thích của bạn', 'phù hợp với bạn'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in personalized_indicators)

    def _extract_recommended_movies_from_response(self, response: str) -> str:
        """Extract information only for movies that are actually recommended in the response."""
        if not response:
            return ""

        # Parse the response to find only the recommended movies
        # Look for numbered lists like "1. Movie Name" or similar patterns
        import re

        # Find movie entries in numbered lists
        movie_entries = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            # Match patterns like "1. Movie Name (year) - genre - description"
            if re.match(r'^\d+\.\s+.+', line):
                movie_entries.append(line)

        if movie_entries:
            # Extract movie titles and basic info from the entries
            movie_titles = []
            for entry in movie_entries[:5]:  # Limit to 5
                # Try to extract the movie title
                title_match = re.search(r'\d+\.\s+([^(]+)', entry)
                if title_match:
                    title = title_match.group(1).strip()
                    movie_titles.append(title)

            # If we have movie titles, try to get their info from database
            if movie_titles and self.analyst and hasattr(self.analyst, 'info_retriever'):
                movie_infos = []
                for title in movie_titles:
                    try:
                        # Search for the movie by title (this might need improvement)
                        # For now, we'll use a simpler approach
                        movie_infos.append(f"Movie: {title}")
                    except:
                        continue
                return "\n".join(movie_infos)

        # Fallback: extract all item IDs mentioned
        item_ids = re.findall(r'\b(\d{1,4})\b', response)
        if item_ids and self.analyst and hasattr(self.analyst, 'info_retriever'):
            movie_infos = []
            for item_id in item_ids[:5]:
                try:
                    info = self.analyst.info_retriever.item_info(item_id=int(item_id))
                    movie_infos.append(info)
                except:
                    continue
            return "\n".join(movie_infos)

        return ""

    def _extract_movie_info_from_response(self, response: str) -> str:
        """Extract movie information from Manager's response for summarization (legacy method)."""
        return self._extract_recommended_movies_from_response(response)

    def _analyze_user_genres(self, user_id: int) -> str:
        """
        Analyze user's preferred genres based on their history using InfoDatabase and Manager LLM.
        """
        try:
            # 1. Get user history
            retriever = None
            if self.analyst and hasattr(self.analyst, 'interaction_retriever'):
                retriever = self.analyst.interaction_retriever
            elif 'Analyst' in self.agents:
                retriever = self.agents['Analyst'].interaction_retriever
            
            if not retriever:
                return ""
                
            # Use user_retrieve_data to get list of (item_id, rating, timestamp)
            try:
                # k=10 items
                history_data = retriever.user_retrieve_data(user_id=user_id, k=10)
            except Exception:
                # Fallback if user_retrieve_data not available or fails
                return ""
            
            if not history_data:
                return ""

            # 2. Resolve items to details (Title, Genre)
            history_details = []
            info_retriever = None
            if self.analyst and hasattr(self.analyst, 'info_retriever'):
                info_retriever = self.analyst.info_retriever
            elif 'Analyst' in self.agents:
                info_retriever = self.agents['Analyst'].info_retriever
                
            for item_id, rating, _ in history_data:
                if info_retriever:
                    # info string usually looks like "Item X Attributes: Title: ...; Genres: ..."
                    info = info_retriever.item_info(item_id=item_id)
                    history_details.append(f"- Movie: {info} (User Rating: {rating})")
                else:
                    history_details.append(f"- Item ID: {item_id} (User Rating: {rating})")

            history_str = "\n".join(history_details)
            
            # 3. Ask LLM to analyze
            prompt = (
                f"Based on the following movies the user watched and rated:\n{history_str}\n\n"
                "Please identify the user's preferred genres and explain briefly why you think so based on the content (plot/themes) of these movies. "
                "Keep it concise (2-3 sentences)."
            )
            analysis = self.manager.thought_llm(prompt)
            return f"\n\n**User Taste Analysis**:\n{analysis}"
        except Exception as e:
            logger.error(f"Error analyzing user genres: {e}")
            return ""

