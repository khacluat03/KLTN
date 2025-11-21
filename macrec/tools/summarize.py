from transformers import pipeline, AutoTokenizer
from transformers.pipelines import SummarizationPipeline

from macrec.llms import AnyOpenAILLM, OpenSourceLLM
from macrec.llms.gemini import AnyGeminiLLM
from macrec.tools.base import Tool
from macrec.utils import get_rm

ALLOWED_GENERATE_KWARGS = {
    'max_length',
    'min_length',
    'max_new_tokens',
    'num_beams',
    'num_return_sequences',
    'temperature',
    'top_k',
    'top_p',
    'length_penalty',
    'repetition_penalty',
    'no_repeat_ngram_size',
    'early_stopping',
}

ALLOWED_PIPELINE_KWARGS = {
    'device',
    'device_map',
    'torch_dtype',
    'trust_remote_code',
    'framework',
    'revision',
    'cache_dir',
    'use_auth_token',
    'model_kwargs',
}

DEFAULT_SUMMARY_PROMPT = (
    "You are a helpful assistant. Summarize the following text in a concise manner.\n\n"
    "Text:\n{text}\n\nSummary:"
)


class TextSummarizer(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.summary_prompt = self.config.get('prompt_template', DEFAULT_SUMMARY_PROMPT)
        self.backend = self._detect_backend()
        if self.backend == 'hf':
            self._init_hf_pipeline()
        else:
            self._init_llm_backend()

    def _detect_backend(self) -> str:
        model_type = self.config.get('model_type')
        if model_type and model_type.lower() != 'huggingface':
            return 'llm'
        if 'model_path' in self.config or 'generate_kwargs' in self.config:
            return 'hf'
        return 'llm' if model_type else 'hf'

    def _init_hf_pipeline(self) -> None:
        self.model_path: str = get_rm(self.config, 'model_path', 't5-base')
        self.model_max_length: int = get_rm(self.config, 'model_max_length', 512)
        raw_generate_kwargs: dict = get_rm(self.config, 'generate_kwargs', {})
        self.generate_kwargs = {
            k: v for k, v in raw_generate_kwargs.items() if k in ALLOWED_GENERATE_KWARGS
        }
        reserved_keys = {'model_path', 'model_max_length', 'generate_kwargs', 'prompt_template'}
        raw_pipeline_kwargs = {
            k: v for k, v in self.config.items() if k not in reserved_keys
        }
        self.pipeline_kwargs = {
            k: v for k, v in raw_pipeline_kwargs.items() if k in ALLOWED_PIPELINE_KWARGS
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, model_max_length=self.model_max_length
        )
        self.pipe: SummarizationPipeline = pipeline(
            'summarization',
            model=self.model_path,
            tokenizer=self.tokenizer,
            **self.pipeline_kwargs,
        )

    def _init_llm_backend(self) -> None:
        llm_config = self.config.copy()
        llm_config.pop('prompt_template', None)
        llm_config.pop('model_path', None)
        llm_config.pop('model_max_length', None)
        llm_config.pop('generate_kwargs', None)
        model_type = llm_config.pop('model_type', 'openai').lower()
        if model_type == 'gemini':
            self.llm = AnyGeminiLLM(**llm_config)
        elif model_type == 'opensource':
            self.llm = OpenSourceLLM(**llm_config)
        else:
            self.llm = AnyOpenAILLM(**llm_config)
        self.pipe = None

    def reset(self) -> None:
        pass

    def summarize(self, text: str) -> str:
        if self.backend == 'hf' and self.pipe is not None:
            summary_text = self.pipe(text, **self.generate_kwargs)[0]['summary_text']
        else:
            prompt = self.summary_prompt.format(text=text)
            summary_text = self.llm(prompt)
        return f"Summarized text: {summary_text}"
