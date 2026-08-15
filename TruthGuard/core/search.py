# search.py - Web search module using DuckDuckGo

from typing import Optional, Any
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from core.utils import clean_snippet


class SearchProvider:
    """Base class for search providers."""
    
    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search the web and return results."""
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider."""
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "snippet": clean_snippet(result.get("body", "")),
                        "url": result.get("href", "")
                    })
                
                return formatted_results
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo search failed: {str(e)}")


def get_search_provider(provider: str) -> SearchProvider:
    """Factory function to get the appropriate search provider."""
    providers = {
        "duckduckgo": DuckDuckGoProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown search provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider]()


def generate_search_query(prompt: str, response: str) -> str:
    """Generate a concise search query from prompt and response."""
    # Extract key entities and claims from the response
    # For simplicity, we combine prompt and response and truncate
    combined = f"{prompt} {response}"
    
    # Remove common filler words and create a focused query
    filler_words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "also", "now", "that", "this",
        "these", "those", "what", "which", "who", "whom", "whose",
        "it", "its", "about", "answer", "question", "response"
    ]
    
    words = combined.lower().split()
    key_words = [w for w in words if w not in filler_words and len(w) > 2]
    
    # Take the most relevant words (up to 10)
    query = " ".join(key_words[:10])
    
    return query.strip() or prompt[:100]
