# main.py - FastAPI application

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    VerifyRequest,
    GenerateAndVerifyRequest,
    VerifyResponse,
    GenerateAndVerifyResponse,
    HealthResponse,
    ErrorResponse,
    VerificationResult,
    GroundingEvidence
)
from core.config import get_settings, Settings
from core.pipeline import TruthGuardPipeline, create_pipeline


# Create FastAPI app
app = FastAPI(
    title="TruthGuard API",
    description="Heterogeneous Hallucination Detection & Grounding Framework for LLM Responses",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: TruthGuardPipeline = None


def get_pipeline() -> TruthGuardPipeline:
    """Get or create the global pipeline instance."""
    global pipeline
    if pipeline is None:
        settings = get_settings()
        
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            missing = ", ".join(validation["missing_keys"])
            raise RuntimeError(f"Missing required API keys: {missing}")
        
        pipeline = create_pipeline(settings)
    return pipeline


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to TruthGuard API",
        "description": "Heterogeneous Hallucination Detection & Grounding Framework",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This help message",
            "GET /health": "Health check",
            "POST /verify": "Verify an existing LLM response",
            "POST /generate-and-verify": "Generate and verify a response"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Try to get pipeline to verify it's working
        get_pipeline()
        return HealthResponse(status="healthy", backend="TruthGuard API", version="1.0.0")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/verify", response_model=VerifyResponse, tags=["Verification"])
async def verify_response(request: VerifyRequest):
    """
    Verify an existing LLM response for hallucination.
    
    - **prompt**: The user's original prompt
    - **response**: The LLM-generated response to verify
    - **regenerate**: Whether to regenerate a safer response if hallucination is detected
    """
    try:
        pipe = get_pipeline()
        result = pipe.verify(
            prompt=request.prompt,
            response=request.response,
            regenerate=request.regenerate
        )
        
        # Convert dict to proper schema objects
        grounding_evidence = None
        if result.get("grounding_evidence"):
            ge = result["grounding_evidence"]
            grounding_evidence = GroundingEvidence(
                query=ge.get("query", ""),
                snippets=ge.get("snippets", []),
                sources=ge.get("sources", []),
                titles=ge.get("titles", []),
                supported=ge.get("supported", False),
                contradicted=ge.get("contradicted", False),
                insufficient=ge.get("insufficient", True)
            )
        
        verification_result = VerificationResult(
            prompt=result["prompt"],
            response=result["response"],
            truth_score=result["truth_score"],
            confidence_score=result["confidence_score"],
            hallucination_probability=result["hallucination_probability"],
            risk_level=result["risk_level"],
            explanation=result["explanation"],
            module_scores=result.get("module_scores"),
            grounding_evidence=grounding_evidence,
            regenerated_response=result.get("regenerated_response"),
            errors=result.get("errors", [])
        )
        
        return VerifyResponse(
            success=True,
            result=verification_result,
            message="Verification completed successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-and-verify", response_model=GenerateAndVerifyResponse, tags=["Verification"])
async def generate_and_verify(request: GenerateAndVerifyRequest):
    """
    Generate a response using the configured base model and verify it.
    
    - **prompt**: The user's prompt
    - **regenerate**: Whether to regenerate a safer response if hallucination is detected
    """
    try:
        pipe = get_pipeline()
        result = pipe.generate_and_verify(
            prompt=request.prompt,
            regenerate=request.regenerate
        )
        
        if result.get("error"):
            return GenerateAndVerifyResponse(
                success=False,
                prompt=request.prompt,
                generated_response="",
                verification=None,
                error=result["error"],
                message="Generation failed"
            )
        
        verification = result.get("verification")
        verification_result = None
        
        if verification:
            grounding_evidence = None
            if verification.get("grounding_evidence"):
                ge = verification["grounding_evidence"]
                grounding_evidence = GroundingEvidence(
                    query=ge.get("query", ""),
                    snippets=ge.get("snippets", []),
                    sources=ge.get("sources", []),
                    titles=ge.get("titles", []),
                    supported=ge.get("supported", False),
                    contradicted=ge.get("contradicted", False),
                    insufficient=ge.get("insufficient", True)
                )
            
            verification_result = VerificationResult(
                prompt=verification["prompt"],
                response=verification["response"],
                truth_score=verification["truth_score"],
                confidence_score=verification["confidence_score"],
                hallucination_probability=verification["hallucination_probability"],
                risk_level=verification["risk_level"],
                explanation=verification["explanation"],
                module_scores=verification.get("module_scores"),
                grounding_evidence=grounding_evidence,
                regenerated_response=verification.get("regenerated_response"),
                errors=verification.get("errors", [])
            )
        
        return GenerateAndVerifyResponse(
            success=True,
            prompt=request.prompt,
            generated_response=result["generated_response"],
            verification=verification_result,
            error=None,
            message="Generation and verification completed"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
