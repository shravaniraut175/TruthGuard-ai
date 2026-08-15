# schemas.py - Pydantic schemas for API

from typing import Optional, Any
from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    """Request schema for /verify endpoint."""
    prompt: str = Field(..., description="User prompt")
    response: str = Field(..., description="LLM-generated response to verify")
    regenerate: bool = Field(default=False, description="Whether to regenerate if hallucination is detected")


class GenerateAndVerifyRequest(BaseModel):
    """Request schema for /generate-and-verify endpoint."""
    prompt: str = Field(..., description="User prompt")
    regenerate: bool = Field(default=False, description="Whether to regenerate if hallucination is detected")


class ModuleScores(BaseModel):
    """Schema for module-wise scores."""
    blackbox: Optional[float] = Field(None, description="Black-box consistency score")
    whitebox: Optional[float] = Field(None, description="White-box token confidence score")
    judge: Optional[float] = Field(None, description="LLM Judge score")
    grounding: Optional[float] = Field(None, description="External grounding score")


class GroundingEvidence(BaseModel):
    """Schema for grounding evidence."""
    query: str = Field(default="", description="Search query used")
    snippets: list[str] = Field(default_factory=list, description="Evidence snippets")
    sources: list[str] = Field(default_factory=list, description="Source URLs")
    titles: list[str] = Field(default_factory=list, description="Result titles")
    supported: bool = Field(default=False, description="Whether evidence supports the response")
    contradicted: bool = Field(default=False, description="Whether evidence contradicts the response")
    insufficient: bool = Field(default=True, description="Whether evidence is insufficient")


class VerificationResult(BaseModel):
    """Schema for verification result."""
    prompt: str
    response: str
    truth_score: float = Field(..., ge=0.0, le=1.0, description="Overall truth score (0-1)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the assessment (0-1)")
    hallucination_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of hallucination (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MODERATE, or HIGH")
    explanation: str = Field(default="", description="Detailed explanation of the assessment")
    module_scores: Optional[dict[str, Optional[float]]] = Field(None, description="Individual module scores")
    grounding_evidence: Optional[GroundingEvidence] = Field(None, description="External grounding evidence")
    regenerated_response: Optional[str] = Field(None, description="Regenerated safer response if applicable")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")


class VerifyResponse(BaseModel):
    """Response schema for /verify endpoint."""
    success: bool = True
    result: VerificationResult
    message: str = "Verification completed successfully"


class GenerateAndVerifyResponse(BaseModel):
    """Response schema for /generate-and-verify endpoint."""
    success: bool = True
    prompt: str
    generated_response: str
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None
    message: str = "Generation and verification completed"


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str = "healthy"
    backend: str = "TruthGuard API"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    success: bool = False
    error: str
    detail: Optional[str] = None
