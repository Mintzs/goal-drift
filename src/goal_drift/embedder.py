from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)