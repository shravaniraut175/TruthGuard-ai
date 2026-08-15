# grounding.py - External grounding module

from typing import Any, Optional
import json

from core.search import SearchProvider, generate_search_query
from core.llm import LLMProvider
from core.utils import clean_snippet, truncate_text


class GroundingModule:
    """External grounding module using web search."""
    
    def __init__(self, search_provider: SearchProvider, judge_provider: LLMProvider):
        self.search_provider = search_provider
        self.judge_provider = judge_provider
    
    def generate_search_query(self, prompt: str, response: str) -> str:
        """Generate a concise search query from prompt and response."""
        return generate_search_query(prompt, response)
    
    def search_evidence(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Search for evidence related to the query."""
        try:
            results = self.search_provider.search(query, max_results=max_results)
            
            if not results:
                return {
                    "query": query,
                    "snippets": [],
                    "sources": [],
                    "error": "No search results found"
                }
            
            snippets = [r["snippet"] for r in results]
            sources = [r["url"] for r in results]
            
            return {
                "query": query,
                "snippets": snippets,
                "sources": sources,
                "titles": [r["title"] for r in results],
                "error": None
            }
        
        except Exception as e:
            return {
                "query": query,
                "snippets": [],
                "sources": [],
                "error": str(e)
            }
    
    def evaluate_evidence(self, prompt: str, response: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Ask the judge model whether evidence supports, contradicts, or is insufficient."""
        snippets = evidence.get("snippets", [])
        
        if not snippets:
            return {
                "supported": False,
                "contradicted": False,
                "insufficient": True,
                "score": 0.5,
                "explanation": "No evidence available for evaluation",
                "error": "No evidence"
            }
        
        # Create evaluation prompt
        evidence_text = "\n\n".join([f"Evidence {i+1}: {s}" for i, s in enumerate(snippets)])
        
        eval_prompt = f"""You are an expert fact-checker. Evaluate whether the following LLM response is supported by, contradicted by, or has insufficient evidence from the provided search results.

PROMPT: {prompt}

LLM RESPONSE: {response}

SEARCH EVIDENCE:
{evidence_text}

Respond with a JSON object in this exact format:
{{
    "supported": true/false,
    "contradicted": true/false,
    "insufficient": true/false,
    "grounding_score": 0.0-1.0,
    "explanation": "Brief explanation of your evaluation"
}}

Rules:
- supported=true if evidence confirms the key claims in the response
- contradicted=true if evidence directly contradicts key claims
- insufficient=true if evidence doesn't clearly support or contradict
- grounding_score should be high (0.8-1.0) if well-supported, low (0.0-0.3) if contradicted
- Only one of supported/contradicted/insufficient should be true
- If both supported and contradicted have some evidence, choose the dominant one"""

        try:
            judge_response = self.judge_provider.generate(eval_prompt, temperature=0.3)
            
            # Parse JSON response
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(judge_response)
                
                return {
                    "supported": result.get("supported", False),
                    "contradicted": result.get("contradicted", False),
                    "insufficient": result.get("insufficient", True),
                    "score": float(result.get("grounding_score", 0.5)),
                    "explanation": result.get("explanation", ""),
                    "error": None
                }
            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback: parse response manually
                response_lower = judge_response.lower()
                supported = "supported" in response_lower and "true" in response_lower
                contradicted = "contradicted" in response_lower and "true" in response_lower
                insufficient = "insufficient" in response_lower and "true" in response_lower
                
                # Default fallback
                return {
                    "supported": supported,
                    "contradicted": contradicted,
                    "insufficient": insufficient or (not supported and not contradicted),
                    "score": 0.5,
                    "explanation": "Could not parse judge response as JSON. Manual parsing attempted.",
                    "error": f"JSON parsing failed: {str(e)}"
                }
        
        except Exception as e:
            return {
                "supported": False,
                "contradicted": False,
                "insufficient": True,
                "score": 0.5,
                "explanation": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }
    
    def verify(self, prompt: str, response: str, max_results: int = 5) -> dict[str, Any]:
        """Full verification pipeline: search and evaluate."""
        try:
            # Generate search query
            query = self.generate_search_query(prompt, response)
            
            # Search for evidence
            evidence = self.search_evidence(query, max_results=max_results)
            
            if evidence.get("error") or not evidence.get("snippets"):
                return {
                    "query": query,
                    "snippets": evidence.get("snippets", []),
                    "sources": evidence.get("sources", []),
                    "supported": False,
                    "contradicted": False,
                    "insufficient": True,
                    "score": 0.5,
                    "explanation": evidence.get("error", "No evidence found"),
                    "error": evidence.get("error")
                }
            
            # Evaluate evidence
            evaluation = self.evaluate_evidence(prompt, response, evidence)
            
            return {
                "query": query,
                "snippets": evidence["snippets"],
                "sources": evidence["sources"],
                "titles": evidence.get("titles", []),
                "supported": evaluation["supported"],
                "contradicted": evaluation["contradicted"],
                "insufficient": evaluation["insufficient"],
                "score": max(0.0, min(1.0, evaluation["score"])),
                "explanation": evaluation["explanation"],
                "error": evaluation.get("error")
            }
        
        except Exception as e:
            return {
                "query": "",
                "snippets": [],
                "sources": [],
                "supported": False,
                "contradicted": False,
                "insufficient": True,
                "score": 0.5,
                "explanation": f"Grounding verification failed: {str(e)}",
                "error": str(e)
            }
