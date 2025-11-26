"""
Synthesizer Agent
Kết hợp outputs từ multiple agents thành response coherent
"""

from typing import Any, List, Dict, Optional
from loguru import logger
import json

from macrec.agents.base import ToolAgent
from macrec.utils import read_json, get_rm, parse_action


class SynthesizerAgent(ToolAgent):
    """
    Agent tổng hợp và kết hợp outputs từ multiple recommendation agents
    Tạo response tự nhiên và coherent
    """

    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 5)
        self.synthesizer = self.get_LLM(config=config)
        self.json_mode = self.synthesizer.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {}

    @property
    def synthesizer_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['synthesizer_prompt_json']
        else:
            return self.prompts['synthesizer_prompt']

    @property
    def synthesizer_examples(self) -> str:
        if self.json_mode:
            return self.prompts['synthesizer_examples_json']
        else:
            return self.prompts['synthesizer_examples']

    @property
    def synthesizer_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['synthesizer_fewshot_json']
        else:
            return self.prompts['synthesizer_fewshot']

    @property
    def hint(self) -> str:
        if 'synthesizer_hint' not in self.prompts:
            return ''
        return self.prompts['synthesizer_hint']

    def _build_synthesizer_prompt(self, **kwargs) -> str:
        return self.synthesizer_prompt.format(
            examples=self.synthesizer_examples,
            fewshot=self.synthesizer_fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )

    def _prompt_synthesizer(self, **kwargs) -> str:
        synthesizer_prompt = self._build_synthesizer_prompt(**kwargs)
        command = self.synthesizer(synthesizer_prompt)
        return command

    def _deduplicate_recommendations(self, all_recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations and rank by confidence"""
        seen_items = set()
        unique_recs = []

        for rec in all_recommendations:
            item_id = rec.get('item_id')
            if item_id and item_id not in seen_items:
                seen_items.add(item_id)
                unique_recs.append(rec)
            elif not item_id:
                # Keep non-item recommendations (like ratings)
                unique_recs.append(rec)

        # Sort by confidence score
        unique_recs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return unique_recs

    def _rank_by_diversity(self, recommendations: List[Dict]) -> List[Dict]:
        """Rank recommendations for diversity (different genres/styles)"""
        if not recommendations:
            return recommendations

        # Simple diversity ranking - prefer different genres
        seen_genres = set()
        diverse_recs = []

        for rec in recommendations:
            genre = rec.get('genre', '').split('|')[0]  # Take first genre
            if genre not in seen_genres or len(seen_genres) >= 3:  # Allow some duplicates
                diverse_recs.append(rec)
                seen_genres.add(genre)

        return diverse_recs

    def synthesize_recommendations(self, agent_outputs: Dict[str, Any],
                                 user_id: Optional[int] = None,
                                 query_type: str = "general") -> Dict[str, Any]:
        """
        Main synthesis method for recommendation outputs
        """
        try:
            # Extract recommendations from different agent outputs
            all_recommendations = []
            metadata = {
                'agents_used': list(agent_outputs.keys()),
                'query_type': query_type,
                'user_id': user_id
            }

            for agent_name, output in agent_outputs.items():
                if isinstance(output, str):
                    # Parse string output to extract recommendations
                    recs = self._parse_agent_output(agent_name, output)
                    all_recommendations.extend(recs)
                elif isinstance(output, list):
                    all_recommendations.extend(output)
                elif isinstance(output, dict) and 'recommendations' in output:
                    all_recommendations.extend(output['recommendations'])

            # Deduplicate and rank
            unique_recs = self._deduplicate_recommendations(all_recommendations)
            final_recs = self._rank_by_diversity(unique_recs)

            # Limit to reasonable number
            final_recs = final_recs[:10]  # Max 10 recommendations

            result = {
                'recommendations': final_recs,
                'metadata': metadata,
                'total_unique': len(final_recs),
                'synthesis_method': 'deduplication + diversity_ranking'
            }

            return result

        except Exception as e:
            logger.error(f"Error in synthesize_recommendations: {e}")
            return {
                'error': str(e),
                'recommendations': [],
                'metadata': {'agents_used': list(agent_outputs.keys())}
            }

    def _parse_agent_output(self, agent_name: str, output: str) -> List[Dict]:
        """Parse string output from agents to extract recommendations"""
        recommendations = []

        try:
            # Parse based on agent type
            if 'sequential' in agent_name.lower():
                # Sequential agent output
                lines = output.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['predict', 'next', 'sequential']):
                        # Extract item info (simplified parsing)
                        rec = {
                            'source': 'sequential',
                            'description': line.strip(),
                            'confidence': 0.7
                        }
                        recommendations.append(rec)

            elif 'cf' in agent_name.lower() or 'collaborative' in agent_name.lower():
                # CF agent output
                lines = output.split('\n')
                for line in lines:
                    if ':' in line and any(char.isdigit() for char in line):
                        # Looks like item: title format
                        rec = {
                            'source': 'collaborative_filtering',
                            'description': line.strip(),
                            'confidence': 0.8
                        }
                        recommendations.append(rec)

            elif 'analyst' in agent_name.lower():
                # Analyst output - usually ratings or analysis
                if 'rating' in output.lower() or any(char.isdigit() for char in output):
                    rec = {
                        'source': 'analyst',
                        'description': output.strip(),
                        'type': 'rating' if 'rating' in output.lower() else 'analysis',
                        'confidence': 0.9
                    }
                    recommendations.append(rec)

        except Exception as e:
            logger.warning(f"Error parsing {agent_name} output: {e}")

        return recommendations

    def generate_natural_response(self, synthesis_result: Dict[str, Any],
                                original_query: str) -> str:
        """
        Generate natural language response from synthesis result
        """
        try:
            recommendations = synthesis_result.get('recommendations', [])
            metadata = synthesis_result.get('metadata', {})
            agents_used = metadata.get('agents_used', [])

            if not recommendations:
                return "I couldn't find any suitable recommendations based on the available data."

            # Build natural response
            response_parts = []

            # Introduction based on agents used
            if len(agents_used) > 1:
                agent_names = []
                for agent in agents_used:
                    if 'sequential' in agent.lower():
                        agent_names.append("sequential pattern analysis")
                    elif 'cf' in agent.lower() or 'collaborative' in agent.lower():
                        agent_names.append("similar user preferences")
                    elif 'analyst' in agent.lower():
                        agent_names.append("personalized analysis")
                    elif 'search' in agent.lower():
                        agent_names.append("database search")

                intro = f"Based on {', '.join(agent_names)}, here are my recommendations:"
                response_parts.append(intro)
            else:
                response_parts.append("Here are my recommendations:")

            # List recommendations
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5
                desc = rec.get('description', 'Unknown item')
                confidence = rec.get('confidence', 0)
                source = rec.get('source', 'unknown')

                # Format based on type
                if rec.get('type') == 'rating':
                    response_parts.append(f"{i}. {desc}")
                else:
                    response_parts.append(f"{i}. {desc}")

            # Add explanation if multiple sources
            if len(agents_used) > 1:
                response_parts.append(f"\nThese recommendations combine insights from {len(agents_used)} different analysis methods for better accuracy.")

            return '\n'.join(response_parts)

        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            return "I have some recommendations but encountered an issue formatting them. Please try rephrasing your query."

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)

        if action_type.lower() == 'synthesize' or action_type.lower() == 'combine':
            # Synthesize multiple agent outputs
            try:
                if self.json_mode:
                    # argument should be dict with agent_outputs
                    agent_outputs = argument.get('agent_outputs', {})
                    user_id = argument.get('user_id')
                    query_type = argument.get('query_type', 'general')
                else:
                    # Parse string argument (simplified)
                    observation = "Synthesize command received. Use JSON mode for complex synthesis."
                    log_head = f':violet[Synthesis requested]:violet[...]\n- '

                synthesis_result = self.synthesize_recommendations(
                    agent_outputs=agent_outputs,
                    user_id=user_id,
                    query_type=query_type
                )

                natural_response = self.generate_natural_response(
                    synthesis_result,
                    original_query=f"Synthesis for user {user_id}"
                )

                observation = f"Synthesis completed. Generated response: {natural_response[:100]}..."

            except Exception as e:
                observation = f"Error in synthesis: {e}"

        elif action_type.lower() == 'format_response':
            # Format response in natural language
            try:
                if self.json_mode:
                    synthesis_result = argument.get('synthesis_result', {})
                    original_query = argument.get('original_query', '')
                else:
                    observation = "Format response command received."

                natural_response = self.generate_natural_response(
                    synthesis_result,
                    original_query
                )

                observation = f"Response formatted: {natural_response[:100]}..."

            except Exception as e:
                observation = f"Error formatting response: {e}"

        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
            log_head = ':violet[Finish with results]:\n- '
        else:
            observation = f'Unknown command type: {action_type}. Valid: synthesize, format_response, finish.'

        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def forward(self, agent_outputs: Dict[str, Any],
                user_id: Optional[int] = None,
                query_type: str = "general") -> str:
        """
        Main synthesis workflow
        """
        while not self.is_finished():
            command = self._prompt_synthesizer(
                agent_outputs=agent_outputs,
                user_id=user_id,
                query_type=query_type
            )
            self.command(command)

        if not self.finished:
            return 'Synthesizer did not complete synthesis.'
        return f'Synthesized recommendations: {self.results}'

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Handle manager calls for synthesis
        """
        if json_mode:
            if not isinstance(argument, dict):
                return 'Invalid argument type for synthesizer: expected dict.'
            agent_outputs = argument.get('agent_outputs', {})
            user_id = argument.get('user_id')
            query_type = argument.get('query_type', 'general')
        else:
            return 'Synthesizer requires JSON mode for complex arguments.'

        return self(agent_outputs=agent_outputs, user_id=user_id, query_type=query_type)


if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    synthesizer = SynthesizerAgent(
        config_path='config/agents/synthesizer_agent.json',
        prompts=read_prompts('config/prompts/agent_prompt/synthesizer.json')
    )

    # Test with sample outputs
    sample_outputs = {
        'sequential': "Predicted next movies: Movie A, Movie B, Movie C",
        'cf': "CF Recommendations: Movie D, Movie E, Movie F",
        'analyst': "User prefers comedy movies"
    }

    result = synthesizer(agent_outputs=sample_outputs, user_id=1)
    print(result)
