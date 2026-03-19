from abc import ABC, abstractmethod

class EmbeddingPort(ABC):
    """
    Abstract port for generating embeddings and calculating similarity.
    """
    @abstractmethod
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates cosine similarity between two text strings.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """
        Generates a numerical embedding vector for the given text.
        """
        pass
