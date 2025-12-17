# Description: This file contains functions for parsing agent actions and answers.

import re
import json
from typing import Any

def parse_action(action: str, json_mode: bool = False) -> tuple[str, Any]:
    """Parse agent action.

    Args:
        `action` (`str`): Agent action in string format.
        `json_mode` (`bool`, optional): Whether the action is in JSON format. Defaults to `False`.
    Returns:
        `tuple[str, Any]`: Action type and argument.
    """
    if json_mode:
        try:
            json_action = json.loads(action)
            return json_action['type'], json_action['content']
        except Exception:
            # Fallback to text parsing if JSON fails (e.g. model outputs Thought trace + Action)
            pass
    if not isinstance(action, str):
        return 'Invalid', None
    action = action.strip()
    
    # Priority 1: Standard Start-Anchored Match (ActionType[Content])
    pattern_std = r'^(\w+)\[(.*)\]'
    match = re.match(pattern_std, action, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
        
    # Priority 2: Finish Action anywhere in the string (Robustness for long context)
    # Often the model writes "Thought: ... Action: Finish[...]" in one block
    if 'Finish[' in action:
        # Find the LAST occurrence of Finish[ to capture the action
        # Allow MISSING closing bracket (truncation)
        pattern_finish = r'(Finish)\[(.*)' 
        matches = re.findall(pattern_finish, action, re.DOTALL)
        if matches:
             # matches is list of tuples (Type, Content). Take the last one.
             action_type, content = matches[-1]
             # Strip trailing bracket if it exists (greedy match might have kept it)
             if content.strip().endswith(']'):
                 content = content.strip()[:-1]
             return action_type, content
    
    # Priority 3: Try to find any ActionType[...] pattern
    # This catches cases where there is prefix text like "Action 1: Search[...]"
    pattern_loose = r'(\w+)\[(.*)\]'
    matches = re.findall(pattern_loose, action, re.DOTALL)
    if matches:
         # Heuristic: Check common action types
         valid_types = ['Search', 'Analyse', 'Finish', 'CF', 'Sequential', 'Reflect']
         for act_type, content in reversed(matches): # Look from end
             if act_type in valid_types:
                 return act_type, content
         
         # If no known type, just return the last match
         return matches[-1][0], matches[-1][1]

    return 'Invalid', None

def parse_raw_answer(answer: str, *args, **kwargs) -> dict[str, bool | str]:
    return {
        'valid': True,
        'answer': answer
    }

def parse_rating_answer(answer: str | int | float, json_mode: bool = False, *args, **kwargs) -> dict[str, float | str]:
    try:
        answer = float(answer)
        if answer < 1 or answer > 5:
            return {
                'valid': False,
                'answer': 0,
                'message': 'Rating should be in range [1, 5].'
            }
    except (ValueError, TypeError):
        return {
            'valid': False,
            'answer': 0,
            'message': 'Rating should be a float number.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': 0,
            'message': 'Other Exception when parsing rating.'
        }
    return {
        'valid': True,
        'answer': answer
    }

def parse_ranking_answer(answer: str | Any, gt_answer: int, n_candidate: int, json_mode: bool = False, *args, **kwargs) -> dict[str, bool | list[int]]:
    if not json_mode:
        candidates = answer.split(',')
    else:
        if isinstance(answer, list):
            candidates = answer
        elif isinstance(answer, str):
            candidates = answer.split(',')
        else:
            return {
                'valid': False,
                'answer': [],
                'message': 'Answer should be a permutated list of candidate ids.'
            }
    try:
        length = len(candidates)
    except TypeError:
        return {
            'valid': False,
            'answer': [],
            'message': 'Answer should be a permutated list of candidate ids.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': [],
            'message': 'Other Exception when parsing ranking answer.'
        }
    if length != n_candidate:
        return {
            'valid': False,
            'answer': [],
            'message': f'Answer should contain {n_candidate} ids, which is the same as the number of candidates in the question.'
        }
    else:
        try:
            answer = [int(c) for c in candidates]
            if gt_answer not in answer:
                return {
                    'valid': False,
                    'answer': [],
                    'message': 'Answer should contain all the candidate ids.'
                }
        except (ValueError, TypeError):
            return {
                'valid': False,
                'answer': [],
                'message': 'The ids in the answer list should be integers.'
            }
    return {
        'valid': True,
        'answer': answer
    }

def parse_answer(type: str, *args, **kwargs) -> dict[str, Any]:
    """Parse answer.

    Args:
        `type` (`str`): Task type. Other arguments are passed to the corresponding parsing function.
    Raises:
        `NotImplementedError`: Unsupported task type.
    Returns:
        `dict[str, Any]`: Parsed answer, including `valid`, `answer`, and `message`. `valid` indicates whether the answer is valid. `answer` is the parsed answer. `message` is the error message if the answer is invalid (otherwise not included).
    """
    if type == 'qa' or type == 'chat' or type == 'gen':
        return parse_raw_answer(*args, **kwargs)
    elif type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported task: {type}')

def init_answer(type: str) -> Any:
    """Initialize answer.

    Args:
        `type` (`str`): Task type.
    Raises:
        `NotImplementedError`: Unsupported task type.
    Returns:
        `Any`: Initialized answer. Different types of answers are returned for different tasks.
    """
    if type == 'qa' or type == 'chat' or type == 'gen':
        return ''
    elif type == 'rp':
        return 0
    elif type == 'sr':
        return []
    else:
        raise NotImplementedError(f'Unsupported task: {type}')
