from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict


def apply_provider_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harmonize per-agent LLM configs with the provider declared in api-config.json.

    We read the provider/model that `init_openai_api` stored in env vars and, when the
    config also targets an API provider (openai/gemini), we rewrite the relevant fields
    so every agent/tool automatically follows the global selection.

    This keeps explicit `model_type: "opensource"` configs untouched.
    """

    if not isinstance(config, dict):
        return config

    provider = os.getenv("MACREC_PROVIDER")
    if not provider:
        return config
    provider = provider.lower()

    model_type = config.get("model_type")
    if model_type not in {"gemini", "openai"}:
        # respect other types such as "opensource"
        return config

    overridden = deepcopy(config)
    default_model = os.getenv("MACREC_MODEL_PATH")

    if provider == "gemini":
        overridden["model_type"] = "gemini"
        overridden["model_name"] = default_model or "gemini-2.0-flash"
    elif provider == "openai":
        overridden["model_type"] = "openai"
        overridden["model_name"] = default_model or "gpt-4o-mini"

    return overridden

