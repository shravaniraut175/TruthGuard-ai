# whitebox.py - White-box token confidence detector

from typing import Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class WhiteBoxDetector:
    """White-box token confidence detector using local model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True
            )
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
    
    def calculate_token_confidence(self, response: str) -> dict[str, Any]:
        """Calculate mean token probability for the response."""
        try:
            self.load_model()
            
            # Tokenize the response
            inputs = self.tokenizer(response, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities for each token
                probs = torch.softmax(logits, dim=-1)
                
                # Get the probability of the actual next token at each position
                token_probs = []
                input_ids = inputs.input_ids[0]
                
                for i in range(len(input_ids) - 1):
                    # Probability of the actual next token
                    next_token_id = input_ids[i + 1]
                    token_prob = probs[0, i, next_token_id].item()
                    token_probs.append(token_prob)
                
                # Calculate mean token probability
                mean_prob = float(torch.mean(torch.tensor(token_probs)).item()) if token_probs else 0.5
                
                return {
                    "score": max(0.0, min(1.0, mean_prob)),
                    "mean_probability": mean_prob,
                    "num_tokens": len(token_probs),
                    "token_probs": token_probs[:10],  # First 10 for debugging
                    "error": None
                }
        
        except Exception as e:
            return {
                "score": 0.5,
                "mean_probability": 0.5,
                "num_tokens": 0,
                "token_probs": [],
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if the white-box model is available."""
        try:
            self.load_model()
            return True
        except Exception:
            return False


# Global white-box detector instance (lazy loaded)
_whitebox_detector: Optional[WhiteBoxDetector] = None


def get_whitebox_detector(model_name: str) -> WhiteBoxDetector:
    """Get or create the global white-box detector."""
    global _whitebox_detector
    if _whitebox_detector is None or _whitebox_detector.model_name != model_name:
        _whitebox_detector = WhiteBoxDetector(model_name)
    return _whitebox_detector


def calculate_whitebox_score(response: str, model_name: str) -> dict[str, Any]:
    """Calculate white-box token confidence score."""
    detector = get_whitebox_detector(model_name)
    return detector.calculate_token_confidence(response)
