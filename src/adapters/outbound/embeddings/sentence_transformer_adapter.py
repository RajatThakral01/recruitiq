import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.ports.embedding_port import EmbeddingPort
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import ScoringException


class SentenceTransformerAdapter(EmbeddingPort):
    """
    Concrete EmbeddingPort implementation using the
    `all-MiniLM-L6-v2` sentence-transformers model.
    """

    def __init__(self) -> None:
        """Load the model once at startup and record loading time."""
        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        start = time.time()
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise ScoringException("Embedding model could not be loaded.", detail=str(e))
        elapsed = round(time.time() - start, 2)
        logger.info(f"Sentence-transformers model loaded in {elapsed}s.")

    def get_embedding(self, text: str) -> list[float]:
        """
        Encode the input text and return a plain Python list of floats.

        Args:
            text: Any string to embed.

        Returns:
            Normalized embedding vector as list[float].
        """
        if not text:
            return []
        try:
            embedding = self.model.encode([text], normalize_embeddings=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ScoringException("Failed to generate embedding.", detail=str(e))

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Returns float in [0, 1]. Empty strings return 0.0.
        """
        if not text1 or not text2:
            return 0.0
        try:
            emb1 = np.array(self.get_embedding(text1))
            emb2 = np.array(self.get_embedding(text2))
            # Since embeddings are normalized, dot product == cosine similarity
            similarity = float(np.dot(emb1, emb2))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise ScoringException("Failed to compute cosine similarity.", detail=str(e))
