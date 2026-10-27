"""Pydantic schemas for LawUP API requests and responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ClauseRequest(BaseModel):
    clause: str


class ContractTextRequest(BaseModel):
    contract_text: str
    max_clauses: Optional[int] = 50


class SimplifyResponse(BaseModel):
    rewrite: str
    latency_ms: float


class AnalyzeResponse(BaseModel):
    original_clause: str
    simplified_rewrite: str
    risks_identified: List[Dict[str, Any]]
    grounding_references: List[Dict[str, Any]]
    verifier_report: Dict[str, Any]
    citation_report: Optional[Dict[str, Any]] = None
    total_latency_ms: float
    stage_latencies_ms: Dict[str, float]


class IngestResponse(BaseModel):
    total_clauses: int
    clauses: List[Dict[str, Any]]


class DocumentAnalyzeResponse(BaseModel):
    total_clauses: int
    clauses: List[Dict[str, Any]]
    total_latency_ms: float
    risk_summary: Dict[str, int]
    verification_summary: Dict[str, int]
