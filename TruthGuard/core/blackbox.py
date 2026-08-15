# blackbox.py - Black-box consistency detector

from typing import Any, Optional
import numpy as np

from core.llm import LLMProvider
from core.embeddings import calculate_batch_similarity


class BlackBoxDetector:
    """Black-box consistency detector using multiple samples."""
    
    def __init__(self, llm_provider: LLMProvider, num_samples: int = 3):
        self.llm_provider = llm_provider
        self.num_samples = num_samples
        # Different temperatures for diversity
        self.temperatures = [0.3, 0.7, 1.0][:num_samples]
    
    def generate_samples(self, prompt: str) -> list[str]:
        """Generate multiple samples from the base model."""
        samples = []
        
        for temp in self.temperatures:
            try:
                response = self.llm_provider.generate(prompt, temperature=temp)
                if response.strip():
                    samples.append(response.strip())
            except Exception as e:
                # Continue with available samples
                pass
        
        return samples
    
    def calculate_consistency_score(self, original_response: str, prompt: str) -> dict[str, Any]:
        """Calculate consistency score between original response and generated samples."""
        try:
            # Generate samples using the same prompt
            samples = self.generate_samples(prompt)
            
            if not samples:
                return {
                    "score": 0.5,  # Default when no samples generated
                    "samples": [],
                    "similarities": [],
                    "error": "No samples generated"
                }
            
            # Calculate semantic similarity between original and samples
            similarities = calculate_batch_similarity(original_response, samples)
            
            # Mean similarity is the consistency score
            mean_similarity = float(np.mean(similarities)) if similarities else 0.5
            
            return {
                "score": max(0.0, min(1.0, mean_similarity)),
                "samples": samples,
                "similarities": similarities,
                "num_samples": len(samples),
                "error": None
            }
        
        except Exception as e:
            return {
                "score": 0.5,
                "samples": [],
                "similarities": [],
                "error": str(e)
            }
