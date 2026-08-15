# regeneration.py - Response regeneration module

from typing import Any, Optional

from core.llm import LLMProvider


class RegenerationModule:
    """Module for regenerating safer responses when hallucination is detected."""
    
    def __init__(self, base_provider: LLMProvider):
        self.base_provider = base_provider
    
    def regenerate(
        self,
        prompt: str,
        original_response: str,
        grounding_evidence: dict[str, Any],
        explanation: str
    ) -> dict[str, Any]:
        """
        Regenerate a safer response using external evidence.
        
        Args:
            prompt: Original user prompt
            original_response: The potentially hallucinated response
            grounding_evidence: Evidence from external search
            explanation: Explanation of why regeneration was triggered
        
        Returns:
            Dictionary with regenerated response and metadata
        """
        snippets = grounding_evidence.get("snippets", [])
        sources = grounding_evidence.get("sources", [])
        
        # Build evidence context
        if snippets:
            evidence_context = "RELEVANT EVIDENCE FROM SEARCH:\n"
            for i, snippet in enumerate(snippets, 1):
                evidence_context += f"{i}. {snippet}\n"
            if sources:
                evidence_context += "\nSOURCES:\n"
                for i, source in enumerate(sources, 1):
                    evidence_context += f"[{i}] {source}\n"
        else:
            evidence_context = "NO RELIABLE EXTERNAL EVIDENCE COULD BE FOUND."
        
        regen_prompt = f"""You are tasked with providing a safe, accurate response to a user query. Your previous response may have contained unverified or potentially inaccurate information.

ORIGINAL PROMPT: {prompt}

PREVIOUS RESPONSE (may contain inaccuracies): {original_response}

ISSUES IDENTIFIED: {explanation}

{evidence_context}

Based on the evidence above, provide a revised response that:
1. Only states information that can be verified from the evidence or is common knowledge
2. Clearly indicates uncertainty where evidence is insufficient
3. Says "I cannot verify this information with confidence" if evidence is lacking
4. Cites sources when making specific claims

Provide your response in this JSON format:
{{
    "response": "Your revised, safer response here",
    "confidence": "high/medium/low",
    "disclaimer": "Any disclaimer about verification status"
}}

Be conservative - it's better to express uncertainty than to potentially mislead."""

        try:
            regenerated = self.base_provider.generate(regen_prompt, temperature=0.3)
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', regenerated, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(regenerated)
            
            return {
                "regenerated_response": result.get("response", regenerated),
                "confidence": result.get("confidence", "medium"),
                "disclaimer": result.get("disclaimer", ""),
                "success": True,
                "error": None
            }
        
        except Exception as e:
            # Fallback: return a generic safe response
            if not snippets:
                safe_response = "I cannot verify this information with confidence based on available evidence. I recommend consulting authoritative sources for accurate information on this topic."
            else:
                safe_response = f"Based on available evidence, here's what can be verified: {snippets[0][:200]}... However, I recommend verifying this information from primary sources."
            
            return {
                "regenerated_response": safe_response,
                "confidence": "low",
                "disclaimer": "This response was auto-generated due to potential accuracy concerns.",
                "success": True,
                "error": f"JSON parsing failed: {str(e)}"
            }
    
    def should_regenerate(
        self,
        hallucination_probability: float,
        threshold: float
    ) -> bool:
        """Determine if regeneration should be triggered."""
        return hallucination_probability >= threshold


def create_regeneration_module(base_provider: LLMProvider) -> RegenerationModule:
    """Create a regeneration module instance."""
    return RegenerationModule(base_provider)
