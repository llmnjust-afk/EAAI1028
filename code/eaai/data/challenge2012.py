"""Causal hourly conversion of official Time,Parameter,Value records."""

import numpy as np
import pandas as pd

STATIC = ("Age", "Gender", "Height", "ICUType")
ALIASES = {
    "SysABP": "SBP", "NISysABP": "SBP", "DiasABP": "DBP",
    "NIDiasABP": "DBP", "NIMAP": "MAP", "RespRate": "Resp",
    "HCT": "Hct", "K": "Potassium", "Mg": "Magnesium",
    "ALP": "Alkalinephos", "Bilirubin": "Bilirubin_total", "TropI": "TroponinI",
}


def hourly_record(frame: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Measurements become available at ceil(timestamp/60), never earlier.

    Output has decision times 0,...,48 hours. Within an hour, the last valid
    measurement wins (including across explicitly mapped sensor aliases).
    No backward filling is performed. This 49-point mortality input is NOT a
    49-hour deterioration follow-up and must not be used for event lead times.
    The source documents nonnegative measurements and -1 as unknown. Other
    negative/nonfinite measurements (including released Temp=-17.8 outliers)
    are marked missing and explicitly counted in the returned QC metadata.
    """
    if list(frame.columns) != ["Time", "Parameter", "Value"] or frame.empty:
        raise ValueError("2012 records require nonempty Time,Parameter,Value columns")
    work = frame.copy()
    clock = work["Time"].astype(str)
    if not clock.str.fullmatch(r"\d{2}:\d{2}").all():
        raise ValueError("invalid 2012 HH:MM timestamp")
    hour = clock.str[:2].astype(int)
    minute = clock.str[3:].astype(int)
    elapsed = hour * 60 + minute
    if (minute >= 60).any() or (elapsed > 48 * 60).any():
        raise ValueError("2012 timestamps must be within the first 48 hours")
    values = pd.to_numeric(work["Value"], errors="raise")
    ids = values[work["Parameter"] == "RecordID"]
    if len(ids) != 1 or ids.iloc[0] != int(patient_id):
        raise ValueError("RecordID must match the patient filename")
    descriptors = work["Parameter"].isin(STATIC)
    if (elapsed[descriptors] != 0).any():
        raise ValueError("admission descriptors must have timestamp 00:00")
    result = pd.DataFrame({"ICULOS": np.arange(49)})
    invalid = (~np.isfinite(values)) | ((values < 0) & (values != -1))
    work["Value"] = values.where(np.isfinite(values) & (values >= 0), np.nan)
    work["elapsed"] = elapsed
    work["bin"] = np.ceil(elapsed / 60).astype(int)
    work["Parameter"] = work["Parameter"].replace(ALIASES)
    for name in STATIC:
        admission = work.loc[work["Parameter"] == name, "Value"].dropna()
        if len(admission) > 1 and admission.nunique() > 1:
            raise ValueError("conflicting admission descriptors")
        result[name] = admission.iloc[0] if len(admission) else np.nan
    dynamic = work[~work["Parameter"].isin((*STATIC, "RecordID"))]
    dynamic = dynamic.dropna(subset=["Value"]).sort_values("elapsed", kind="stable")
    dynamic = dynamic.drop_duplicates(["bin", "Parameter"], keep="last")
    if not dynamic.empty:
        wide = dynamic.pivot(index="bin", columns="Parameter", values="Value")
        result = result.join(wide, how="left")
    result.attrs["invalid_measurements"] = int(invalid.sum())
    return result
