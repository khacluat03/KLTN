import json
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
        db_keywords = [
            # Explicit query intent
            'database', 'bảng', 'table', 'truy vấn', 'query', 'sql',
            'select', 'top', 'liệt kê', 'tìm', 'find', 'search', 'kiếm',
            'danh sách', 'list', 'thống kê', 'count', 'bao nhiêu', 'how many',
            'những ai', 'who', 'cái gì', 'what', 'ở đâu', 'where',
            'tổng hợp', 'summary', 'lọc', 'filter', 'sắp xếp', 'sort',
            'gợi ý', 'suggest', 'recommend', 'tiếp theo', 'next', 'tương tự', 'similar'
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
        if self._is_database_task and not (self._search_performed or self._analyse_performed):
            self.scratchpad += '\n⚠️ CRITICAL: This task requires querying the database or analyzing data. You MUST use Search[...] or Analyse[...] action first. You CANNOT Finish without searching or analyzing first.'
            
            # Specific hint for sequential recommendation
            task_prompt_lower = (self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')).lower()
            if any(k in task_prompt_lower for k in ['gợi ý', 'suggest', 'recommend', 'tiếp theo', 'next', 'tương tự', 'similar']):
                 self.scratchpad += '\n💡 HINT: If the user mentions a specific item (movie, product), you should first Search for its ID (e.g., Search[What is the item_id of movie X?]), then use that ID to Analyse or Search for similar items.'
        
        # NEW: Detect user_id in query and enforce Analyse
        import re
        combined_text = self.manager_kwargs.get('task_prompt', '') + ' ' + self.manager_kwargs.get('input', '')
        user_id_match = re.search(r'user[_\s]?(\d+)', combined_text, re.IGNORECASE)
        if user_id_match and not self._analyse_performed:
            user_id = user_id_match.group(1)
            self.scratchpad += f'\n🚨 MANDATORY: Query mentions user_{user_id}. You MUST call Analyse[user, {user_id}] FIRST to get their profile and history before making any recommendations. This is NOT optional.'
        
        # Check if previous thought mentioned Search but haven't searched yet
        if self.require_search_before_finish and not (self._search_performed or self._analyse_performed):
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
                logger.info(f"AUTO-FINISH: Manager didn't Finish, using CF results")
                action = f"Finish[{auto_response}]"
        
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        
        # Auto-convert Finish to Search for database tasks
        if action_type.lower() == 'finish' and self._is_database_task and not (self._search_performed or self._analyse_performed):
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
            
            if self._is_database_task and not self._search_performed and not has_cf_results:
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
            elif self.require_search_before_finish and not (self._search_performed or self._analyse_performed):
                # Legacy check for non-database tasks (if any)
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
                
                # NEW: Smart Finish for personalized recommendations when Manager fails to Finish properly
                # If we have search results and Manager output is invalid/empty, use search candidates
                if (self._search_performed and 
                    hasattr(self, '_candidates') and 
                    len(self._candidates) >= 5):
                    
                    # Check if argument is invalid or empty
                    if isinstance(argument, str):
                        import re
                        arg_ids = re.findall(r'\d+', argument)
                        
                        # If argument has no IDs or very few, use our candidates
                        if len(arg_ids) < 5:
                            # Use top 5 from candidates
                            top_5_ids = self._candidates[:5]
                            # Get item details
                            item_details = []
                            for item_id in top_5_ids:
                                item_info = self.searcher.info_retriever.item_info(item_id=item_id)
                                item_details.append(f"{item_id}: {item_info}")
                            
                            argument = f"Here are 5 movie recommendations:\n" + "\n".join([f"{i+1}. {detail}" for i, detail in enumerate(item_details)])
                            logger.info(f"Smart Finish: Using search candidates for personalized recommendation")
                            self.scratchpad += f'\n✅ AUTO-ENHANCED: Using top 5 items from Search results'
                
                parse_result = self._parse_answer(argument)
                if parse_result['valid']:
                    observation = self.finish(parse_result['answer'])
                    log_head = ':violet[Finish with answer]:\n- '
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
                self._analyse_performed = True
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
        self.reset(clear=True)
        self.input = user_input
        
        # 1. Interpreter (NLU)
        # Use Interpreter to understand the query and generate a task prompt
        if self.interpreter:
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
                return final_response
            else:
                return str(ranked_result)
        else:
            # Chat/Explain Flow
            if self.interpreter:
                return self.interpreter.invoke(argument=user_input, json_mode=False)
            else:
                return "Interpreter not configured for chat."

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
            response = self(user_input=user_input, reset=True)
            print(f"System: {response}")
