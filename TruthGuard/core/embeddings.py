# embeddings.py - Semantic similarity using sentence-transformers

from typing import Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Sentence transformer for semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def batch_cosine_similarity(self, reference: str, candidates: list[str]) -> list[float]:
        """Calculate cosine similarity between reference and multiple candidates."""
        all_texts = [reference] + candidates
        embeddings = self.encode(all_texts)
        
        reference_embedding = embeddings[0]
        similarities = []
        
        for i in range(1, len(embeddings)):
            similarity = np.dot(reference_embedding, embeddings[i]) / (
                np.linalg.norm(reference_embedding) * np.linalg.norm(embeddings[i])
            )
            similarities.append(float(similarity))
        
        return similarities


# Global embedding model instance (lazy loaded)
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    model = get_embedding_model()
    return model.cosine_similarity(text1, text2)


def calculate_batch_similarity(reference: str, candidates: list[str]) -> list[float]:
    """Calculate semantic similarity between reference and multiple candidates."""
    model = get_embedding_model()
    return model.batch_cosine_similarity(reference, candidates)
