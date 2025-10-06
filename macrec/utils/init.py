# Description: Initialization functions.

import os
import random
import numpy as np
import torch

def init_openai_api(api_config: dict):
    """Initialize OpenAI API (tolerates missing api_base/api_key)."""
    api_base = api_config.get('api_base')
    api_key = api_config.get('api_key')
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

def init_gemini_api(api_config: dict):
    """Initialize Google Gemini API using provided api_key or env.

    Accepts {"api_key": "..."} and sets both GOOGLE_API_KEY and GEMINI_API_KEY.
    """
    import google.generativeai as genai
    api_key = api_config.get('api_key') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Gemini api_config must include 'api_key' or set GOOGLE_API_KEY/GEMINI_API_KEY env.")
    genai.configure(api_key=api_key)
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GEMINI_API_KEY'] = api_key

def init_all_seeds(seed: int = 0) -> None:
    """Initialize all seeds.

    Args:
        `seed` (`int`, optional): Random seed. Defaults to `0`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
