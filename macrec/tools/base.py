from abc import ABC, abstractmethod

from macrec.utils import read_json, apply_provider_overrides

class Tool(ABC):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        self.config = apply_provider_overrides(read_json(config_path))

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("reset method not implemented")

class RetrievalTool(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def search(self, query: str) -> str:
        raise NotImplementedError("search method not implemented")

    @abstractmethod
    def lookup(self, title: str, term: str) -> str:
        raise NotImplementedError("lookup method not implemented")
