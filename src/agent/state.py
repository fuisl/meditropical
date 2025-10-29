from pydantic import BaseModel, Field, constr
from typing import List, Dict, Any, Optional
class DiseaseHypothesis(BaseModel):
    disease: str
    p: float = Field(ge=0, le=1)
    support: List[str] = []
    against: List[str] = []
    evidence_ids: List[str] = []

class CaseState(BaseModel):
    case_id: str
    demographics: Dict[str, Any] = {}
    locale: Dict[str, Any] = {}
    exposures: List[Dict[str, Any]] = []
    symptoms: List[Dict[str, Any]] = []
    vitals_series: List[Dict[str, Any]] = []
    exam: Dict[str, Any] = {}
    labs: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    derived_features: Dict[str, Any] = {}
    ddx: List[DiseaseHypothesis] = []
    planned_actions: List[str] = []
    evidence: List[Dict[str, Any]] = []
    uncertainty: Dict[str, Any] = {}
    provenance: List[Dict[str, Any]] = []
