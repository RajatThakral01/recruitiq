from src.core.ports.embedding_port import EmbeddingPort
from src.infrastructure.logger import logger


class SentenceTransformerAdapter(EmbeddingPort):
    """Lightweight stub for sentence-transformers.
    In llm_first mode this is never called.
    """

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence-transformers model loaded.")
        except ImportError:
            self._model = None
            logger.warning("sentence-transformers not installed — using LLM scoring only.")

    def encode(self, texts: list[str]):
        if self._model is None:
            return [[0.0] * 384] * len(texts)
        return self._model.encode(texts)

    def get_embedding(self, text: str) -> list[float]:
        if self._model is None:
            return [0.0] * 384
        return self._model.encode([text])[0].tolist()

    def get_similarity(self, text1: str, text2: str) -> float:
        if self._model is None:
            return 0.5
        embeddings = self._model.encode([text1, text2])
        a, b = embeddings[0], embeddings[1]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.5
