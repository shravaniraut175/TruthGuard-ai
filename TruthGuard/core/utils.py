# utils.py - Utility functions

import re
import json
from typing import Any, Optional


def extract_json_from_response(response: str) -> Optional[dict[str, Any]]:
    """Extract JSON from a potentially messy LLM response."""
    # Try to find JSON object in the response
    json_pattern = r'\{[^{}]*\}'
    
    # First try direct JSON parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*({.*?})\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON-like object
    matches = re.findall(json_pattern, response, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def normalize_score(score: Optional[float], default: float = 0.5) -> float:
    """Normalize a score to be between 0 and 1."""
    if score is None:
        return default
    return max(0.0, min(1.0, float(score)))


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_snippet(snippet: str) -> str:
    """Clean a search result snippet."""
    # Remove excessive whitespace
    snippet = re.sub(r'\s+', ' ', snippet)
    # Remove special characters that might cause issues
    snippet = snippet.replace('\n', ' ').strip()
    return truncate_text(snippet, 300)


def format_sources(sources: list[str]) -> list[dict[str, str]]:
    """Format source URLs into a structured format."""
    formatted = []
    for i, url in enumerate(sources, 1):
        formatted.append({
            "index": i,
            "url": url,
            "display": f"[{i}] {url}"
        })
    return formatted


def calculate_risk_level(hallucination_probability: float) -> str:
    """Calculate risk level based on hallucination probability."""
    if hallucination_probability < 0.30:
        return "LOW"
    elif hallucination_probability < 0.60:
        return "MODERATE"
    else:
        return "HIGH"


def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level."""
    colors = {
        "LOW": "#22c55e",      # Green
        "MODERATE": "#f59e0b", # Amber
        "HIGH": "#dc2626"      # Red
    }
    return colors.get(risk_level, "#6b7280")
