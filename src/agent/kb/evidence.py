# kb/evidence.py
from pydantic import BaseModel
from typing import List, Optional

class EvidenceItem(BaseModel):
    id: str
    text: str
    source_path: Optional[str] = None
    span: Optional[str] = None
    score: Optional[float] = None

class EvidencePack(BaseModel):
    query: str
    items: List[EvidenceItem]
