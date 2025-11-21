import os
import random
import numpy as np
import torch


def init_openai_api(api_config: dict):
    """
    Initialize API environment variables.
    Works for:
      - OpenAI (if provider == 'openai')
      - Gemini (if provider == 'gemini')
    """

    provider = api_config.get('provider')
    api_key = api_config.get('api_key')
    model_path = api_config.get('model_path')
    api_base = api_config.get('api_base')

    # ---- CASE 1: OPENAI PROVIDER ----
    if provider == "openai":
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    # ---- CASE 2: GEMINI PROVIDER ----
    elif provider == "gemini":
        # Không đặt OPENAI_API_KEY (tránh lỗi)
        if api_base:
            os.environ["GEMINI_API_BASE"] = api_base

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

    # Store provider + model globally
    if provider:
        os.environ["MACREC_PROVIDER"] = provider
    if model_path:
        os.environ["MACREC_MODEL_PATH"] = model_path


def init_gemini_api(api_config: dict):
    """
    Initialize Google Gemini API using provided api_key or env.
    """
    import google.generativeai as genai

    api_key = (
        api_config.get('api_key') or
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('GEMINI_API_KEY')
    )

    if not api_key:
        raise ValueError("Gemini API key missing: set api_key or GOOGLE_API_KEY/GEMINI_API_KEY.")

    genai.configure(api_key=api_key)

    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GEMINI_API_KEY'] = api_key


def init_all_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
