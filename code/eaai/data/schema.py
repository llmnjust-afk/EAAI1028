"""Variable schemas for the two Challenge cohorts."""

from __future__ import annotations

CHALLENGE_2019_VARIABLES: tuple[str, ...] = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
)

CHALLENGE_2019_LABEL = "SepsisLabel"

CHALLENGE_2019_STATIC: tuple[str, ...] = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")

CHALLENGE_2019_DYNAMIC: tuple[str, ...] = tuple(
    v for v in CHALLENGE_2019_VARIABLES if v not in CHALLENGE_2019_STATIC and v != "ICULOS"
)

CHALLENGE_2012_VARIABLES: tuple[str, ...] = (
    "HR", "SBP", "Temp", "Resp", "WBC", "Creatinine", "Glucose", "Lactate",
    "Hct", "Platelets",
)

HOURS_PER_DAY = 24.0
