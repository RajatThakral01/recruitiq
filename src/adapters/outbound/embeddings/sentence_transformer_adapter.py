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
            logger.warning("sentence-transformers not installed — semantic scoring disabled. Using LLM scoring only.")

    def encode(self, texts: list[str]):
        if self._model is None:
            return []
        return self._model.encode(texts)
