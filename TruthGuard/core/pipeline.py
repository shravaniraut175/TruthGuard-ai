# pipeline.py - Main verification pipeline

from typing import Any, Optional

from core.config import Settings
from core.llm import create_base_provider, create_judge_provider
from core.search import get_search_provider
from core.blackbox import BlackBoxDetector
from core.whitebox import WhiteBoxDetector
from core.grounding import GroundingModule
from core.judge import JudgeModule
from core.fusion import ScoreFusion
from core.regeneration import RegenerationModule
from core.utils import calculate_risk_level


class TruthGuardPipeline:
    """Main pipeline for hallucination detection and verification."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize providers
        self.base_provider = create_base_provider(settings)
        self.judge_provider = create_judge_provider(settings)
        
        # Initialize search provider
        self.search_provider = get_search_provider(settings.search_provider)
        
        # Initialize detectors
        self.blackbox_detector = BlackBoxDetector(
            self.base_provider,
            settings.num_blackbox_samples
        )
        
        # White-box detector (optional)
        self.whitebox_detector = None
        self.whitebox_available = False
        if settings.whitebox_model:
            try:
                self.whitebox_detector = WhiteBoxDetector(settings.whitebox_model)
                self.whitebox_available = True
            except Exception:
                self.whitebox_available = False
        
        # Initialize modules
        self.grounding_module = GroundingModule(self.search_provider, self.judge_provider)
        self.judge_module = JudgeModule(self.judge_provider)
        
        # Initialize score fusion with normalized weights
        weights = {
            "blackbox": settings.blackbox_weight,
            "whitebox": settings.whitebox_weight if self.whitebox_available else 0.0,
            "judge": settings.judge_weight,
            "grounding": settings.grounding_weight
        }
        self.score_fusion = ScoreFusion(weights)
        
        # Initialize regeneration module
        self.regeneration_module = RegenerationModule(self.base_provider)
    
    def verify(
        self,
        prompt: str,
        response: str,
        regenerate: bool = False
    ) -> dict[str, Any]:
        """
        Verify an LLM response for hallucination.
        
        Args:
            prompt: User prompt
            response: LLM-generated response to verify
            regenerate: Whether to regenerate if hallucination probability is high
        
        Returns:
            Verification result dictionary
        """
        result = {
            "prompt": prompt,
            "response": response,
            "truth_score": 0.5,
            "confidence_score": 0.5,
            "hallucination_probability": 0.5,
            "risk_level": "MODERATE",
            "explanation": "",
            "module_scores": {},
            "grounding_evidence": {},
            "regenerated_response": None,
            "errors": []
        }
        
        try:
            # 1. Black-box consistency check
            blackbox_result = self.blackbox_detector.calculate_consistency_score(response, prompt)
            blackbox_score = blackbox_result.get("score", 0.5)
            
            if blackbox_result.get("error"):
                result["errors"].append(f"Blackbox: {blackbox_result['error']}")
            
            # 2. White-box token confidence (if available)
            whitebox_score = None
            if self.whitebox_available and self.whitebox_detector:
                whitebox_result = self.whitebox_detector.calculate_token_confidence(response)
                whitebox_score = whitebox_result.get("score", 0.5)
                
                if whitebox_result.get("error"):
                    result["errors"].append(f"Whitebox: {whitebox_result['error']}")
            
            # 3. LLM-as-a-Judge evaluation
            judge_result = self.judge_module.evaluate(prompt, response)
            judge_score = judge_result.get("overall_score", 0.5)
            judge_factual_accuracy = judge_result.get("factual_accuracy", 0.5)
            judge_explanation = judge_result.get("explanation", "")
            
            if judge_result.get("error"):
                result["errors"].append(f"Judge: {judge_result['error']}")
            
            # 4. External grounding
            grounding_result = self.grounding_module.verify(
                prompt, response, self.settings.max_search_results
            )
            grounding_score = grounding_result.get("score", 0.5)
            grounding_contradicted = grounding_result.get("contradicted", False)
            
            if grounding_result.get("error"):
                result["errors"].append(f"Grounding: {grounding_result['error']}")
            
            # 5. Score fusion
            fusion_result = self.score_fusion.fuse(
                blackbox_score=blackbox_score,
                whitebox_score=whitebox_score,
                judge_score=judge_score,
                grounding_score=grounding_score,
                grounding_contradicted=grounding_contradicted,
                judge_factual_accuracy=judge_factual_accuracy
            )
            
            result["truth_score"] = fusion_result["truth_score"]
            result["confidence_score"] = fusion_result["confidence_score"]
            result["hallucination_probability"] = fusion_result["hallucination_probability"]
            result["module_scores"] = fusion_result["module_scores"]
            
            # 6. Calculate risk level
            result["risk_level"] = calculate_risk_level(result["hallucination_probability"])
            
            # 7. Build explanation
            explanations = []
            if judge_explanation:
                explanations.append(f"Judge Analysis: {judge_explanation}")
            if grounding_result.get("explanation"):
                explanations.append(f"Grounding: {grounding_result['explanation']}")
            if fusion_result.get("veto_applied"):
                explanations.append(f"Vetoes applied: {', '.join(fusion_result['veto_applied'])}")
            
            result["explanation"] = "\n\n".join(explanations) if explanations else "Analysis complete."
            
            # 8. Grounding evidence
            result["grounding_evidence"] = {
                "query": grounding_result.get("query", ""),
                "snippets": grounding_result.get("snippets", []),
                "sources": grounding_result.get("sources", []),
                "titles": grounding_result.get("titles", []),
                "supported": grounding_result.get("supported", False),
                "contradicted": grounding_result.get("contradicted", False),
                "insufficient": grounding_result.get("insufficient", True)
            }
            
            # 9. Regeneration if needed
            if regenerate and self.regeneration_module.should_regenerate(
                result["hallucination_probability"],
                self.settings.hallucination_threshold
            ):
                regen_result = self.regeneration_module.regenerate(
                    prompt,
                    response,
                    result["grounding_evidence"],
                    result["explanation"]
                )
                result["regenerated_response"] = regen_result.get("regenerated_response")
            
            return result
        
        except Exception as e:
            result["errors"].append(f"Pipeline error: {str(e)}")
            result["explanation"] = f"Verification failed: {str(e)}"
            return result
    
    def generate_and_verify(
        self,
        prompt: str,
        regenerate: bool = False
    ) -> dict[str, Any]:
        """
        Generate a response and verify it.
        
        Args:
            prompt: User prompt
            regenerate: Whether to regenerate if hallucination probability is high
        
        Returns:
            Dictionary with generated response and verification result
        """
        try:
            # Generate response using base model
            generated_response = self.base_provider.generate(prompt, temperature=0.7)
            
            # Verify the generated response
            verification_result = self.verify(prompt, generated_response, regenerate)
            
            return {
                "prompt": prompt,
                "generated_response": generated_response,
                "verification": verification_result
            }
        
        except Exception as e:
            return {
                "prompt": prompt,
                "generated_response": "",
                "error": f"Generation failed: {str(e)}",
                "verification": None
            }


def create_pipeline(settings: Settings) -> TruthGuardPipeline:
    """Create a TruthGuard pipeline instance."""
    return TruthGuardPipeline(settings)
