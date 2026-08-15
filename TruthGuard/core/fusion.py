# fusion.py - Score fusion module

from typing import Any, Optional

from core.utils import normalize_score


class ScoreFusion:
    """Score fusion module combining multiple detector scores."""
    
    def __init__(self, weights: dict[str, float]):
        """
        Initialize with weights for each score component.
        
        Args:
            weights: Dictionary with keys 'blackbox', 'whitebox', 'judge', 'grounding'
        """
        self.weights = weights
    
    def fuse(
        self,
        blackbox_score: Optional[float],
        whitebox_score: Optional[float],
        judge_score: Optional[float],
        grounding_score: Optional[float],
        grounding_contradicted: bool = False,
        judge_factual_accuracy: float = 0.5
    ) -> dict[str, Any]:
        """
        Fuse multiple scores into a final truth score.
        
        Args:
            blackbox_score: Score from black-box consistency detector
            whitebox_score: Score from white-box token confidence (can be None)
            judge_score: Overall score from LLM judge
            grounding_score: Score from external grounding
            grounding_contradicted: Whether evidence contradicts the response
            judge_factual_accuracy: Factual accuracy score from judge
        
        Returns:
            Dictionary with final scores and explanations
        """
        # Normalize all scores
        blackbox_score = normalize_score(blackbox_score, 0.5)
        whitebox_score = normalize_score(whitebox_score, 0.5)
        judge_score = normalize_score(judge_score, 0.5)
        grounding_score = normalize_score(grounding_score, 0.5)
        
        # Get normalized weights (handle missing whitebox)
        weights = self._get_active_weights(whitebox_score is not None)
        
        # Calculate weighted average
        components = [
            ("blackbox", blackbox_score, weights.get("blackbox", 0)),
            ("whitebox", whitebox_score, weights.get("whitebox", 0)),
            ("judge", judge_score, weights.get("judge", 0)),
            ("grounding", grounding_score, weights.get("grounding", 0))
        ]
        
        weighted_sum = sum(score * weight for _, score, weight in components)
        total_weight = sum(weight for _, _, weight in components)
        
        if total_weight == 0:
            base_truth_score = 0.5
        else:
            base_truth_score = weighted_sum / total_weight
        
        # Apply vetoes
        final_truth_score = base_truth_score
        veto_applied = []
        
        # External contradiction veto
        if grounding_contradicted and grounding_score < 0.35:
            # Strong reduction for contradicted evidence
            final_truth_score = min(final_truth_score, 0.25)
            veto_applied.append("external_contradiction")
        
        # Judge factual accuracy veto
        if judge_factual_accuracy < 0.20:
            # Strong reduction for low factual accuracy
            final_truth_score = min(final_truth_score, 0.30)
            veto_applied.append("low_factual_accuracy")
        
        # Ensure score is in valid range
        final_truth_score = max(0.0, min(1.0, final_truth_score))
        
        # Calculate confidence score (based on agreement between modules)
        active_scores = [s for s, w in [(blackbox_score, weights.get("blackbox", 0)),
                                         (whitebox_score, weights.get("whitebox", 0)),
                                         (judge_score, weights.get("judge", 0)),
                                         (grounding_score, weights.get("grounding", 0))] 
                        if w > 0]
        
        if len(active_scores) >= 2:
            # Confidence based on variance between scores
            import statistics
            variance = statistics.variance(active_scores) if len(active_scores) > 1 else 0
            confidence_score = max(0.0, min(1.0, 1.0 - variance))
        else:
            confidence_score = 0.5
        
        # Hallucination probability
        hallucination_probability = 1.0 - final_truth_score
        
        return {
            "truth_score": round(final_truth_score, 4),
            "confidence_score": round(confidence_score, 4),
            "hallucination_probability": round(hallucination_probability, 4),
            "base_truth_score": round(base_truth_score, 4),
            "veto_applied": veto_applied,
            "module_scores": {
                "blackbox": round(blackbox_score, 4),
                "whitebox": round(whitebox_score, 4) if whitebox_score is not None else None,
                "judge": round(judge_score, 4),
                "grounding": round(grounding_score, 4)
            },
            "weights_used": weights
        }
    
    def _get_active_weights(self, whitebox_available: bool = True) -> dict[str, float]:
        """Get normalized weights for active modules."""
        weights = {
            "blackbox": self.weights.get("blackbox", 0.25),
            "whitebox": self.weights.get("whitebox", 0.25) if whitebox_available else 0.0,
            "judge": self.weights.get("judge", 0.25),
            "grounding": self.weights.get("grounding", 0.25)
        }
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


def create_score_fusion(weights: dict[str, float]) -> ScoreFusion:
    """Create a score fusion instance."""
    return ScoreFusion(weights)
