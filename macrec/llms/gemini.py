import os
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from macrec.llms.basellm import BaseLLM


class AnyGeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.0-flash", json_mode: bool = False, *args, **kwargs):
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get("max_tokens", 512)
        self.max_context_length: int = 8192

        if json_mode:
            logger.info("Using JSON mode of Gemini API.")
            if "model_kwargs" in kwargs:
                kwargs["model_kwargs"]["response_format"] = {"type": "json_object"}
            else:
                kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        api_key = (
            kwargs.get("google_api_key")
            or kwargs.get("api_key")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            convert_system_message_to_human=True,
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=self.max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ("google_api_key", "api_key", "temperature", "max_tokens", "model_type", "model_name")},
        )
        self.model_type = "chat"

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content.replace("\n", " ").strip()


