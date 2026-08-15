# TruthGuard

**Heterogeneous Hallucination Detection & Grounding Framework for LLM Responses**

TruthGuard is an AI hallucination detection framework that verifies Large Language Model responses before presenting them to users. It combines multiple detection techniques including black-box consistency, white-box token confidence, LLM-as-a-Judge evaluation, and external grounding via web search.

## Features

- **Heterogeneous Model Ensembling**: Configure different models for response generation and judgment (e.g., Qwen for base, Claude for judge)
- **External Search Grounding**: DuckDuckGo search for real-time evidence verification
- **Multi-Detector Fusion**: Combines black-box, white-box, judge, and grounding scores
- **Confidence-Based Regeneration**: Automatically regenerates safer responses when hallucination probability is high
- **Weighted Score Fusion**: Normalized weights with automatic handling of disabled modules
- **Veto Mechanisms**: External contradiction and low factual accuracy vetoes

## Project Structure

```
TruthGuard/
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env.example          # Environment variables template
├── .gitignore
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── schemas.py        # Pydantic schemas
└── core/
    ├── __init__.py
    ├── config.py         # Configuration management
    ├── utils.py          # Utility functions
    ├── llm.py            # LLM provider abstraction
    ├── search.py         # Web search module
    ├── embeddings.py     # Semantic similarity
    ├── blackbox.py       # Black-box consistency detector
    ├── whitebox.py       # White-box token confidence
    ├── grounding.py      # External grounding module
    ├── judge.py          # LLM-as-a-Judge
    ├── fusion.py         # Score fusion
    ├── regeneration.py   # Response regeneration
    └── pipeline.py       # Main verification pipeline
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TruthGuard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment variables and configure:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` file with your API keys and preferences:

- `OPENROUTER_API_KEY`: Required for OpenRouter models (Qwen, Claude, GPT via OpenRouter)
- `GOOGLE_API_KEY`: Required for Gemini models
- `OPENAI_API_KEY`: Optional for direct OpenAI access
- `BASE_PROVIDER`: Provider for base model (openrouter, google, openai)
- `BASE_MODEL`: Model name for generating responses
- `JUDGE_PROVIDER`: Provider for judge model (should differ from base)
- `JUDGE_MODEL`: Model name for evaluation
- `WHITEBOX_MODEL`: Hugging Face model for token confidence (can be empty to disable)
- `HALLUCINATION_THRESHOLD`: Threshold for triggering regeneration (default: 0.60)

## Usage

### Start the FastAPI Backend

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Start the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

The frontend will be available at `http://localhost:8501`

### API Endpoints

#### GET /
Returns welcome message and API info.

#### GET /health
Health check endpoint.

#### POST /verify
Verify an existing LLM response.

Request body:
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "regenerate": false
}
```

#### POST /generate-and-verify
Generate a response and verify it.

Request body:
```json
{
  "prompt": "What is the capital of France?",
  "regenerate": false
}
```

### Response Format

```json
{
  "truth_score": 0.92,
  "confidence_score": 0.88,
  "hallucination_probability": 0.08,
  "risk_level": "LOW",
  "explanation": "Response is factually accurate...",
  "module_scores": {
    "blackbox": 0.95,
    "whitebox": 0.85,
    "judge": 0.90,
    "grounding": 0.95
  },
  "grounding_evidence": {
    "query": "capital of France",
    "snippets": ["Paris is the capital..."],
    "sources": ["https://..."],
    "supported": true,
    "contradicted": false,
    "insufficient": false
  },
  "regenerated_response": null
}
```

## How It Works

### 1. Black-Box Consistency
Generates multiple samples from the base model at different temperatures and measures semantic similarity with the original response.

### 2. White-Box Token Confidence
Uses a local Hugging Face model to calculate mean token probability for the response tokens.

### 3. LLM-as-a-Judge
A different model evaluates factual accuracy, grounding, coherence, and provides an explanation.

### 4. External Grounding
Searches the web for evidence and verifies if the response is supported or contradicted.

### 5. Score Fusion
Combines all scores with configurable weights, applies veto mechanisms for contradictions.

### 6. Regeneration
If hallucination probability exceeds threshold, generates a safer response using evidence.

## Deployment

### Render (Backend)

1. Create a new Web Service on Render
2. Connect your repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from `.env.example`

### Streamlit Cloud (Frontend)

1. Create a new app on Streamlit Cloud
2. Connect your repository
3. Set the main file path to `streamlit_app.py`
4. Add secrets from `.env.example`

## License

MIT License
