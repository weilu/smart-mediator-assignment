from typing import Dict, List, Tuple, TypedDict

MediatorId = int
CaseId = int
CourtStationId = str
CaseTypeId = str

CaseLoads = Dict[MediatorId, int]
MediatorVAs = Dict[MediatorId, float]
CaseHistory = Tuple[List[float], List[float]]

class MediatorBeliefDict(TypedDict):
    mu: float
    sd: float

MediatorInitVA = Dict[MediatorId, MediatorBeliefDict]
MedByCrtCaseType = Dict[CourtStationId, Dict[CaseTypeId, List[MediatorId]]]
AvgCaseRate = Dict[CaseTypeId, Dict[CourtStationId, float]]
AvgPValByCrtCaseType = Dict[Tuple[CaseTypeId, CourtStationId], float]
