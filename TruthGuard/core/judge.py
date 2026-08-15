# judge.py - LLM-as-a-Judge module

from typing import Any, Optional
import json
import re

from core.llm import LLMProvider
from core.utils import extract_json_from_response


class JudgeModule:
    """LLM-as-a-Judge module for evaluating response quality."""
    
    def __init__(self, judge_provider: LLMProvider):
        self.judge_provider = judge_provider
    
    def evaluate(self, prompt: str, response: str) -> dict[str, Any]:
        """Evaluate the response using the judge model."""
        
        eval_prompt = f"""You are an expert evaluator of LLM responses. Your task is to assess the quality and factual accuracy of a response to a given prompt.

PROMPT: {prompt}

LLM RESPONSE: {response}

Evaluate the response on the following criteria:
1. Factual Accuracy: Are the claims in the response factually correct?
2. Grounding: Is the response well-grounded in verifiable information?
3. Coherence: Is the response logically coherent and well-structured?
4. Completeness: Does the response adequately address the prompt?

Respond with a JSON object in this exact format:
{{
    "factual_accuracy": 0.0-1.0,
    "grounding": 0.0-1.0,
    "coherence": 0.0-1.0,
    "completeness": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "explanation": "Detailed explanation of your evaluation, highlighting any factual errors, unsupported claims, or logical inconsistencies"
}}

Scoring guidelines:
- factual_accuracy: 1.0 if all claims are verifiably true, 0.0 if demonstrably false
- grounding: 1.0 if well-supported by common knowledge or verifiable sources
- coherence: 1.0 if logically structured with no contradictions
- completeness: 1.0 if fully addresses all aspects of the prompt
- overall_score: weighted average reflecting the most critical issues

Be strict about factual accuracy. If you detect any hallucinations or fabricated information, score accordingly."""

        try:
            judge_response = self.judge_provider.generate(eval_prompt, temperature=0.3)
            
            # Parse JSON response
            result = extract_json_from_response(judge_response)
            
            if result is None:
                return self._fallback_evaluation(judge_response)
            
            return {
                "factual_accuracy": float(result.get("factual_accuracy", 0.5)),
                "grounding": float(result.get("grounding", 0.5)),
                "coherence": float(result.get("coherence", 0.5)),
                "completeness": float(result.get("completeness", 0.5)),
                "overall_score": float(result.get("overall_score", 0.5)),
                "explanation": result.get("explanation", "No explanation provided"),
                "raw_response": judge_response,
                "error": None
            }
        
        except Exception as e:
            return {
                "factual_accuracy": 0.5,
                "grounding": 0.5,
                "coherence": 0.5,
                "completeness": 0.5,
                "overall_score": 0.5,
                "explanation": f"Evaluation failed: {str(e)}",
                "raw_response": "",
                "error": str(e)
            }
    
    def _fallback_evaluation(self, raw_response: str) -> dict[str, Any]:
        """Fallback evaluation when JSON parsing fails."""
        # Try to extract scores from text
        score_pattern = r'(\d+\.?\d*)\s*/\s*(?:1\.?0?|10)'
        matches = re.findall(score_pattern, raw_response)
        
        if matches:
            scores = [float(m) / 10 if float(m) > 1 else float(m) for m in matches]
            avg_score = sum(scores) / len(scores) if scores else 0.5
        else:
            avg_score = 0.5
        
        return {
            "factual_accuracy": avg_score,
            "grounding": avg_score,
            "coherence": avg_score,
            "completeness": avg_score,
            "overall_score": avg_score,
            "explanation": "Fallback evaluation: Could not parse structured response. Score based on heuristic analysis.",
            "raw_response": raw_response,
            "error": "JSON parsing failed, using fallback"
        }


def create_judge_module(judge_provider: LLMProvider) -> JudgeModule:
    """Create a judge module instance."""
    return JudgeModule(judge_provider)
