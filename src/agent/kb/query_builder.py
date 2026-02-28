# kb/query_builder.py
from typing import List, Dict, Any

def build_queries_from_state(state: Dict[str, Any]) -> List[str]:
    geo = state.get("locale", {})
    region = geo.get("country") or geo.get("region") or "tropics"
    day = state.get("derived_features", {}).get("day_of_illness", None)

    ddx = [h["disease"] for h in state.get("ddx", [])][:3] or []
    sxs = [s.get("name") for s in state.get("symptoms", [])]

    base = []
    if "fever" in sxs:
        base.append(f"{region} undifferentiated febrile illness algorithm adult")

    # dengue
    if any("dengue" in d.lower() for d in ddx) or "retro-orbital pain" in sxs:
        when = f" day {day}" if day else ""
        base += [
            f"WHO dengue guideline{when} warning signs triage fluid management",
            "dengue NS1 vs IgM timing interpretation outpatient vs inpatient"
        ]

    # malaria
    if any("malaria" in d.lower() for d in ddx):
        base += [
            "malaria treatment severe vs uncomplicated dosing artesunate ACT",
            "thick thin smear percent parasitemia grading severe criteria"
        ]

    return list(dict.fromkeys(base))  # de-dup while preserving order
