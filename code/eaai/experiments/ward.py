"""Ward-level translation of per-patient alarm rates.

A per-patient rate of 1/day is difficult to interpret operationally.  This
module replays observed per-hour alarm traces on a synthetic ward with Poisson
arrivals, finite beds and a fixed alarm-handling capacity.  It reports the
fraction of hours in which the ward demand exceeds capacity, the peak and 95th
percentile hourly alarm counts, occupancy and refused admissions.
"""

from __future__ import annotations

import numpy as np


def simulate_ward(fired, lengths, arrival_rate: float = 5.0,
                  horizon_days: float = 30.0, n_beds: int = 20,
                  capacity_per_hour: float = 2.0, warmup_days: float = 7.0,
                  seed: int = 0) -> dict:
    """Simulate a ward by resampling observed patient alarm traces.

    ``fired`` is a list of boolean per-hour alarm traces and ``lengths`` the
    observed length of each trace in hours.  Arrivals are Poisson with
    ``arrival_rate`` patients per day.  A patient is admitted only when a bed is
    free; otherwise the admission is refused.  Alarms start at the hour after
    admission, matching the one-step-ahead deployment convention.
    """
    fired = [np.asarray(trace, dtype=bool) for trace in fired]
    lengths = np.asarray(lengths, dtype=int)
    if not fired:
        raise ValueError("at least one trace is required")
    if np.any(lengths <= 0) or np.any(lengths > np.array([f.size for f in fired])):
        raise ValueError("lengths must be positive and within the traces")
    if arrival_rate < 0 or n_beds <= 0 or capacity_per_hour <= 0:
        raise ValueError("arrival_rate, n_beds and capacity must be positive")
    if horizon_days <= warmup_days:
        raise ValueError("horizon must exceed the warmup period")

    horizon = int(round(horizon_days * 24))
    warmup = int(round(warmup_days * 24))
    max_stay = int(lengths.max())
    occupancy = np.zeros(horizon + max_stay + 2, dtype=int)
    alarms = np.zeros(horizon + max_stay + 2, dtype=float)
    rng = np.random.default_rng(seed)
    admitted = 0
    refused = 0

    for hour in range(horizon):
        arrivals = rng.poisson(arrival_rate / 24.0)
        for _ in range(arrivals):
            candidates = np.flatnonzero(np.asarray(lengths) <= max_stay)
            if not candidates.size:
                refused += 1
                continue
            patient = int(rng.choice(candidates))
            stay = int(lengths[patient])
            free = np.flatnonzero(occupancy[hour:hour + stay] < n_beds)
            if free.size < stay:
                refused += 1
                continue
            occupancy[hour:hour + stay] += 1
            trace = fired[patient][:stay]
            alarms[hour + 1:hour + 1 + stay] += trace.astype(float)
            admitted += 1

    window = slice(warmup, horizon)
    occupied = occupancy[window]
    demand = alarms[window]
    over = demand > capacity_per_hour
    return {
        "admitted": int(admitted),
        "refused": int(refused),
        "refused_fraction": float(refused / (admitted + refused))
        if admitted + refused else float("nan"),
        "mean_occupancy": float(occupied.mean()),
        "bed_utilization": float(occupied.mean() / n_beds),
        "alarm_rate_per_bed_day": float(demand.sum() / max(occupied.sum(), 1) * 24.0),
        "peak_hourly_alarms": float(demand.max()),
        "p95_hourly_alarms": float(np.percentile(demand, 95)),
        "fraction_hours_over_capacity": float(over.mean()),
        "capacity_per_hour": float(capacity_per_hour),
        "n_beds": int(n_beds),
        "arrival_rate": float(arrival_rate),
        "horizon_days": float(horizon_days),
        "warmup_days": float(warmup_days),
    }
