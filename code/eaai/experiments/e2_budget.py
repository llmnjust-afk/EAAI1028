"""Matched-budget comparison and the lead-time/budget Pareto frontier.

Loss matrices are computed once per score cache and policy configuration, then
reused across calibration/test partitions, budgets and seeds. This is required
for cohort-scale experiments: recomputing every patient's stateful trajectory
for every budget is unnecessarily expensive.
"""

from __future__ import annotations

import numpy as np

from ..policy.alarm import HOURS_PER_DAY, AlarmPolicy, max_alarms_per_day, run_policy
from ..policy.conformal import BlockedConformalCalibrator, TimestepConformalCalibrator
from ..policy.selection import (
    CRCSelector, LTTSelector, PluginSelector, RandomizedMixtureSelector,
    select_by_utility, select_by_validation)
from ..eval.utility import prediction_utility_rows


def _validate_sequences(scores, onset, lengths, step_hours=1.0):
    lengths = np.asarray(lengths)
    onset = np.asarray(onset)
    n = len(scores)
    if (not n or lengths.shape != (n,) or onset.shape != (n,) or
            not np.isfinite(lengths).all() or np.any(lengths <= 0) or
            np.any(lengths != np.floor(lengths)) or not np.isfinite(onset).all() or
            np.any(onset < -1) or np.any(onset != np.floor(onset))):
        raise ValueError("positive integer lengths and integer onsets (-1 for none) required")
    if not np.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("sampling interval must be finite and positive")
    lengths, onset = lengths.astype(int), onset.astype(int)
    sequences = []
    for i, seq in enumerate(scores):
        seq = np.asarray(seq, dtype=float)
        if seq.ndim != 1 or seq.size < lengths[i]:
            raise ValueError("score sequences must cover each declared follow-up")
        seq = seq[:lengths[i]]
        if np.isinf(seq).any():
            raise ValueError("valid scores must be finite or NaN (missing)")
        sequences.append(seq)
    return sequences, onset, lengths


def _validate_partitions(n, calibration, test):
    partitions = []
    for indices in (calibration, test):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not indices.size or not np.isfinite(indices).all() or
                np.any(indices != np.floor(indices)) or np.any(indices < 0) or
                np.any(indices >= n) or np.unique(indices).size != indices.size):
            raise ValueError("partition indices must be nonempty, unique and in range")
        partitions.append(indices.astype(int))
    calibration, test = partitions
    if np.intersect1d(calibration, test).size:
        raise ValueError("calibration and test patients must be disjoint")
    return calibration, test


def alarm_loss_matrix(scores, lengths, thresholds, refractory: int = 4,
                      escalation: int = 0, step_hours: float = 1.0) -> np.ndarray:
    """Per-patient alarms/day at every threshold, shape ``(patients, thresholds)``.

    The result depends only on the score cache and the fixed policy family, so it
    must be computed once and reused across budgets and repeated partitions.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if (thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all()
            or not np.all(np.diff(thresholds) > 0)):
        raise ValueError("threshold grid must be nonempty, finite and strictly increasing")
    sequences, _, lengths = _validate_sequences(scores, np.zeros(len(scores)), lengths,
                                                step_hours=step_hours)
    n = len(sequences)
    width = max((seq.size for seq in sequences), default=0)
    padded = np.full((n, width), np.nan)
    for i, seq in enumerate(sequences):
        padded[i, :seq.size] = seq
    per_day = np.zeros((n, thresholds.size))
    for k, threshold in enumerate(thresholds):
        exceeded = np.isfinite(padded) & (padded >= float(threshold))
        fired = np.zeros_like(exceeded, dtype=bool)
        refractory_left = np.zeros(n, dtype=int)
        streak = np.zeros(n, dtype=int)
        for t in range(width):
            hit = exceeded[:, t]
            streak = np.where(hit, streak + 1, 0)
            override = (escalation > 0) & (streak >= escalation)
            fire = hit & ((refractory_left == 0) | override)
            fired[:, t] = fire
            refractory_left = np.where(fire, refractory,
                                       np.maximum(refractory_left - 1, 0))
            streak = np.where(fire, 0, streak)
        per_day[:, k] = (fired.sum(axis=1) * HOURS_PER_DAY /
                         (lengths * step_hours))
    return per_day


def patient_policy_outcomes(scores, onset, lengths, threshold: float,
                           refractory: int = 4, escalation: int = 0,
                           lead_window_hours: float = 12.0,
                           step_hours: float = 1.0) -> dict:
    """Per-patient alarm rate, event detection and lead time at one threshold.

    Event detection is defined only for subjects with an onset inside observed
    follow-up; the remaining entries are NaN and are excluded from sensitivity.
    """
    sequences, onset, lengths = _validate_sequences(scores, onset, lengths,
                                                    step_hours=step_hours)
    horizon = int(np.floor(lead_window_hours / step_hours))
    rates = np.empty(len(sequences), dtype=float)
    detected = np.full(len(sequences), np.nan, dtype=float)
    leads = np.full(len(sequences), np.nan, dtype=float)
    for i, seq in enumerate(sequences):
        fired = run_policy(seq[:lengths[i]], AlarmPolicy(threshold, refractory, escalation))
        rates[i] = float(np.count_nonzero(fired)) * HOURS_PER_DAY / (lengths[i] * step_hours)
        t_event = int(onset[i])
        if t_event <= 0 or t_event >= lengths[i]:
            continue
        start = max(t_event - horizon, 0)
        hit = np.flatnonzero(fired[start:t_event])
        detected[i] = float(hit.size > 0)
        if hit.size:
            leads[i] = float((t_event - (start + hit[0])) * step_hours)
    return {"alarm_rate": rates, "detected": detected, "lead_time": leads,
            "event": np.isfinite(detected)}

def patient_utility_matrix(scores, onset, lengths, thresholds: np.ndarray,
                           refractory: int = 4, escalation: int = 0,
                           lead_hours: int = 6, step_hours: float = 1.0) -> np.ndarray:
    """Per-patient normalised utility at every candidate threshold.

    The challenge labels are reconstructed from the event onset using the same
    six-hour lead as the 2019 cohort.  Utility is additive over patients, which
    lets the selector use a patient-blocked standard error without treating
    hourly rows as independent observations.
    """
    sequences, onset, lengths = _validate_sequences(
        scores, onset, lengths, step_hours=step_hours)
    thresholds = np.asarray(thresholds, dtype=float)
    if (thresholds.ndim != 1 or not thresholds.size or
            not np.isfinite(thresholds).all() or
            not np.all(np.diff(thresholds) > 0)):
        raise ValueError("threshold grid must be nonempty, finite and strictly increasing")
    n = len(sequences)
    width = max((seq.size for seq in sequences), default=0)
    labels = np.zeros((n, width), dtype=float)
    for i, event in enumerate(onset):
        event = int(event)
        if 0 <= event < lengths[i]:
            labels[i, max(0, event - int(lead_hours)):lengths[i]] = 1.0

    predictions = np.zeros((n, width, thresholds.size), dtype=bool)
    for i, seq in enumerate(sequences):
        for k, threshold in enumerate(thresholds):
            fired = run_policy(seq[:lengths[i]],
                               AlarmPolicy(float(threshold), refractory, escalation))
            predictions[i, :lengths[i], k] = fired

    zeros = np.zeros_like(labels)
    inactive = prediction_utility_rows(labels, zeros)
    optimal = zeros.copy()
    for i, event in enumerate(onset):
        event = int(event)
        if 0 <= event < lengths[i]:
            optimal[i, max(0, event - int(lead_hours))] = 1.0
    oracle = prediction_utility_rows(labels, optimal)
    denominator = float(oracle.sum() - inactive.sum())
    if denominator <= 0:
        raise ValueError("optimal utility must exceed inactive utility")
    rows = np.empty((n, thresholds.size), dtype=float)
    for k in range(thresholds.size):
        rows[:, k] = (prediction_utility_rows(labels, predictions[:, :, k])
                      - inactive) / denominator
    return rows


def evaluate_threshold(scores, onset, lengths, threshold: float,
                       refractory: int = 4, escalation: int = 0,
                       lead_window_hours: float = 12.0,
                       step_hours: float = 1.0) -> dict:
    """Evaluate a fixed threshold without fitting on a calibration partition."""
    sequences, onset, lengths = _validate_sequences(
        scores, onset, lengths, step_hours=step_hours)
    return _policy_outcomes([sequences[i] for i in range(len(sequences))],
                            onset, lengths, threshold, refractory, escalation,
                            lead_window_hours, step_hours)


def _policy_outcomes(scores: list[np.ndarray], onset: np.ndarray,
                     lengths: np.ndarray, threshold: float,
                     refractory: int, escalation: int,
                     lead_window_hours: float, step_hours: float) -> dict:
    """Per-patient policy outcomes and clinically interpretable alarm metrics.

    ``ppv`` is the fraction of emitted alarms that fall inside an eligible
    event's pre-onset detection window.  Alarms on patients without an
    observable event are false alarms.  ``nna`` (number needed to alert) is the
    reciprocal of PPV when at least one event is detected.
    """
    horizon = int(np.floor(lead_window_hours / step_hours))
    detected, leads, rates = [], [], []
    n_alarms = 0
    n_true_alarms = 0
    patient_days = 0.0

    for i, seq in enumerate(scores):
        fired = run_policy(seq[:lengths[i]], AlarmPolicy(threshold, refractory, escalation))
        hours = lengths[i] * step_hours
        rates.append(float(np.count_nonzero(fired)) * HOURS_PER_DAY / hours)
        n_alarms += int(np.count_nonzero(fired))
        patient_days += hours / HOURS_PER_DAY

        t_event = int(onset[i])
        if t_event <= 0 or t_event >= lengths[i]:
            continue
        start = max(t_event - horizon, 0)
        window = fired[start:t_event]
        n_true_alarms += int(np.count_nonzero(window))
        hit = np.flatnonzero(window)
        if hit.size:
            detected.append(True)
            leads.append((t_event - (start + hit[0])) * step_hours)
        else:
            detected.append(False)

    n_events = len(detected)
    detected_count = int(sum(detected))
    ppv = float(n_true_alarms / n_alarms) if n_alarms else float("nan")
    return {
        "alarm_rate": float(np.mean(rates)),
        "sensitivity": float(np.mean(detected)) if detected else float("nan"),
        "median_lead_time": float(np.median(leads)) if leads else float("nan"),
        "detected": detected_count,
        "n_events": n_events,
        "n_alarms": int(n_alarms),
        "n_true_alarms": int(n_true_alarms),
        "patients": int(len(scores)),
        "patient_days": float(patient_days),
        "alarm_rate_pooled": float(n_alarms / patient_days) if patient_days else float("nan"),
        "ppv": ppv,
        "nna": float(1.0 / ppv) if np.isfinite(ppv) and ppv > 0 else float("nan"),
        "alarms_per_detected_event": (float(n_alarms / detected_count)
                                      if detected_count else float("nan")),
    }


def _mixture_outcomes(scores: list[np.ndarray], onset: np.ndarray,
                      lengths: np.ndarray, low: float, high: float, p_low: float,
                      refractory: int, escalation: int,
                      lead_window_hours: float, step_hours: float) -> dict:
    """Expected outcomes when each patient draws the low threshold with ``p_low``.

    Randomising the threshold once per patient makes the alarm rate linear in
    ``p_low`` and the detection probability a two-point mixture.  The lead-time
    median is computed with the induced two-point weights, which is exact for
    this randomisation scheme.
    """
    low_out = _policy_outcomes(scores, onset, lengths, low, refractory, escalation,
                               lead_window_hours, step_hours)
    high_out = _policy_outcomes(scores, onset, lengths, high, refractory, escalation,
                                lead_window_hours, step_hours)
    p = float(p_low)
    q = 1.0 - p

    def mix(a, b):
        return float(p * a + q * b) if np.isfinite(a) and np.isfinite(b) else float("nan")

    # Detection probabilities and lead-time values per patient for the two
    # deterministic thresholds, used to build the mixture distribution.
    horizon = int(np.floor(lead_window_hours / step_hours))
    leads = []
    weights = []
    for i, seq in enumerate(scores):
        t_event = int(onset[i])
        if t_event <= 0 or t_event >= lengths[i]:
            continue
        for threshold, weight in ((low, p), (high, q)):
            if weight <= 0:
                continue
            fired = run_policy(seq[:lengths[i]], AlarmPolicy(threshold, refractory, escalation))
            start = max(t_event - horizon, 0)
            hit = np.flatnonzero(fired[start:t_event])
            if hit.size:
                leads.append((t_event - (start + hit[0])) * step_hours)
                weights.append(weight)
            else:
                leads.append(np.nan)
                weights.append(weight)
    if leads:
        leads_arr = np.asarray(leads, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        valid = np.isfinite(leads_arr)
        if valid.any():
            order = np.argsort(leads_arr[valid])
            values = leads_arr[valid][order]
            w = weights_arr[valid][order]
            cumulative = np.cumsum(w) / w.sum()
            median_lead = float(values[np.searchsorted(cumulative, 0.5)])
        else:
            median_lead = float("nan")
    else:
        median_lead = float("nan")

    n_alarms = mix(low_out["n_alarms"], high_out["n_alarms"])
    n_true = mix(low_out["n_true_alarms"], high_out["n_true_alarms"])
    ppv = float(n_true / n_alarms) if n_alarms else float("nan")
    detected = mix(low_out["detected"], high_out["detected"])
    return {
        "alarm_rate": mix(low_out["alarm_rate"], high_out["alarm_rate"]),
        "sensitivity": mix(low_out["sensitivity"], high_out["sensitivity"]),
        "median_lead_time": median_lead,
        "detected": detected,
        "n_events": low_out["n_events"],
        "n_alarms": n_alarms,
        "n_true_alarms": n_true,
        "patients": low_out["patients"],
        "patient_days": low_out["patient_days"],
        "alarm_rate_pooled": mix(low_out["alarm_rate_pooled"], high_out["alarm_rate_pooled"]),
        "ppv": ppv,
        "nna": float(1.0 / ppv) if np.isfinite(ppv) and ppv > 0 else float("nan"),
        "alarms_per_detected_event": (float(n_alarms / detected)
                                      if detected else float("nan")),
        "mixed": True,
        "threshold_low": float(low),
        "threshold_high": float(high),
        "probability_low": p,
    }


def evaluate_from_losses(scores, onset, lengths, calibration, test, per_day,
                         thresholds, budget: float, alpha: float = 0.1,
                         refractory: int = 4, escalation: int = 0,
                         lead_window_hours: float = 12.0, step_hours: float = 1.0,
                         slack: str = "bernstein",
                         min_stay_hours: float | None = None,
                         procedure: str = "blocked",
                         validation=None, selection: str = "default",
                         n_probability_points: int = 11,
                         utility_rows: np.ndarray | None = None) -> dict:
    """Calibrate a supplied loss matrix and evaluate a disjoint test partition.

    ``procedure="timestep"`` is an effective-sample-size diagnostic shortcut
    without a patient-level guarantee, included only as a comparator.
    """
    sequences, onset, lengths = _validate_sequences(
        scores, onset, lengths, step_hours=step_hours)
    calibration, test = _validate_partitions(len(sequences), calibration, test)
    per_day = np.asarray(per_day, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if per_day.shape != (len(sequences), thresholds.size):
        raise ValueError("loss matrix shape does not match patients and thresholds")
    if (not np.isfinite(per_day).all() or np.any(per_day < 0)):
        raise ValueError("loss matrix must be finite and nonnegative")

    min_stay_hours = step_hours if min_stay_hours is None else min_stay_hours
    l_max = max_alarms_per_day(refractory, step_hours=step_hours,
                               min_stay_hours=min_stay_hours, escalation=escalation)
    if np.any(lengths * step_hours < min_stay_hours):
        raise ValueError("records violate the independently declared minimum stay")
    if per_day.max() > l_max + 1e-9:
        raise AssertionError("per-patient loss exceeds L_max; declare the minimum stay")

    if selection not in ("default", "validation", "utility"):
        raise ValueError("selection must be 'default', 'validation' or 'utility'")
    if selection == "utility" and validation is None:
        raise ValueError("utility selection requires a disjoint validation partition")
    validation_index = None
    validation_report = None
    val_losses = None
    if validation is not None:
        validation = np.asarray(validation, dtype=int)
        if (validation.ndim != 1 or not validation.size or
                np.any(validation < 0) or np.any(validation >= len(sequences)) or
                np.unique(validation).size != validation.size):
            raise ValueError("validation indices must be nonempty, unique and in range")
        if (np.intersect1d(validation, calibration).size or
                np.intersect1d(validation, test).size):
            raise ValueError("validation must be disjoint from calibration and test")
        val_losses = per_day[validation]

    if selection == "utility" and utility_rows is None:
        utility_rows = patient_utility_matrix(
            [sequences[i] for i in validation], onset[validation],
            lengths[validation], thresholds, refractory=refractory,
            escalation=escalation, step_hours=step_hours)
    if utility_rows is not None:
        utility_rows = np.asarray(utility_rows, dtype=float)
        if utility_rows.shape != (len(validation), thresholds.size):
            raise ValueError("utility rows do not match the validation partition and grid")

    kind = "single"
    low_index = high_index = None
    p_low = 1.0
    if procedure == "timestep":
        calibrator = TimestepConformalCalibrator(budget=budget, alpha=alpha)
        calibrator.fit(per_day[calibration], thresholds,
                       n_steps=int(lengths[calibration].sum()), l_max=l_max)
    elif procedure in ("blocked", "blocked_hoeffding", "blocked_bernstein"):
        kind_slack = ("hoeffding" if procedure == "blocked_hoeffding"
                      else "bernstein" if procedure == "blocked_bernstein" else slack)
        calibrator = BlockedConformalCalibrator(budget=budget, alpha=alpha,
                                                 slack=kind_slack)
        calibrator.fit(per_day[calibration], thresholds, l_max)
    elif procedure in ("plugin", "global"):
        calibrator = PluginSelector(budget=budget, alpha=alpha)
        calibrator.fit(per_day[calibration], thresholds, l_max)
    elif procedure == "crc":
        calibrator = CRCSelector(budget=budget, alpha=alpha)
        calibrator.fit(per_day[calibration], thresholds, l_max)
    elif procedure.startswith("ltt_"):
        parts = procedure.split("_")
        kind_slack = parts[1]
        correction = parts[2] if len(parts) > 2 else "holm"
        if kind_slack not in ("hoeffding", "bernstein"):
            raise ValueError("LTT slack must be hoeffding or bernstein")
        if correction not in ("holm", "bonferroni"):
            raise ValueError("LTT correction must be holm or bonferroni")
        calibrator = LTTSelector(budget=budget, alpha=alpha, slack=kind_slack,
                                 correction=correction)
        calibrator.fit(per_day[calibration], thresholds, l_max)
    elif procedure.startswith("randomized"):
        kind_slack = "hoeffding" if procedure.endswith("hoeffding") else slack
        calibrator = RandomizedMixtureSelector(
            budget=budget, alpha=alpha, slack=kind_slack,
            n_probability_points=n_probability_points)
        calibrator.fit(per_day[calibration], thresholds, l_max)
        if calibrator.diagnostics_["feasible"]:
            kind = "mixture"
            low_index = int(calibrator.diagnostics_["index"])
            high_index = int(calibrator.diagnostics_["index_high"])
            p_low = float(calibrator.diagnostics_["probability_low"])
    else:
        raise ValueError(
            "procedure must be blocked, blocked_hoeffding, blocked_bernstein, "
            "plugin, global, crc, ltt_{hoeffding|bernstein}_{holm|bonferroni}, "
            "randomized, randomized_hoeffding or timestep")

    if selection in ("validation", "utility") and kind == "mixture":
        raise ValueError("validation/utility selection is not defined for randomised mixtures")
    if (selection == "validation" and val_losses is not None
            and kind == "single" and calibrator.diagnostics_.get("feasible")):
        validation_report = select_by_validation(
            calibrator, val_losses, thresholds, budget=budget,
            validation_scores=[sequences[i] for i in validation],
            validation_onset=onset[validation],
            validation_lengths=lengths[validation],
            refractory=refractory, escalation=escalation,
            lead_window_hours=lead_window_hours)
        validation_index = int(validation_report["index"])
        if validation_index >= 0:
            calibrator.diagnostics_["index"] = validation_index
            calibrator.diagnostics_["empirical_risk"] = float(
                per_day[calibration, validation_index].mean())
            calibrator.threshold_ = float(thresholds[validation_index])
            calibrator.diagnostics_["status"] = "validation_selected"
    if (selection == "utility" and val_losses is not None
            and kind == "single" and calibrator.diagnostics_.get("feasible")):
        validation_report = select_by_utility(
            calibrator, val_losses, thresholds, budget=budget,
            utility_rows=utility_rows)
        validation_index = int(validation_report["index"])
        if validation_index >= 0:
            calibrator.diagnostics_["index"] = validation_index
            calibrator.diagnostics_["empirical_risk"] = float(
                per_day[calibration, validation_index].mean())
            calibrator.threshold_ = float(thresholds[validation_index])
            calibrator.diagnostics_["status"] = "utility_selected"
            calibrator.diagnostics_["n_feasible"] = int(
                validation_report["n_feasible"])

    raw_threshold = calibrator.threshold_
    if raw_threshold is None:
        raise RuntimeError("calibrator did not produce a threshold")
    index = int(calibrator.diagnostics_.get("index", -1))
    feasible = bool(calibrator.diagnostics_.get("feasible", index >= 0))
    if kind == "mixture" and feasible:
        threshold = float(calibrator.diagnostics_["threshold_low"])
        outcomes = _mixture_outcomes(
            [sequences[i] for i in test], onset[test], lengths[test],
            low=float(calibrator.diagnostics_["threshold_low"]),
            high=float(calibrator.diagnostics_["threshold_high"]),
            p_low=float(calibrator.diagnostics_["probability_low"]),
            refractory=refractory, escalation=escalation,
            lead_window_hours=lead_window_hours, step_hours=step_hours)
        test_losses = (p_low * per_day[test, low_index]
                       + (1.0 - p_low) * per_day[test, high_index])
    else:
        threshold = float(raw_threshold)
        outcomes = _policy_outcomes(
            [sequences[i] for i in test], onset[test], lengths[test],
            threshold, refractory, escalation,
            lead_window_hours, step_hours)
        test_losses = (per_day[test, index] if feasible else np.zeros(test.size))

    outcomes.update({
        "budget": float(budget),
        "threshold": threshold,
        "patient_violation": float(np.mean(test_losses > budget)),
        "certified_risk": calibrator.diagnostics_["certified_risk"],
        "status": calibrator.diagnostics_.get("status", "grid_feasible"),
        "procedure": procedure,
        "selection": selection,
        "n_feasible": int(calibrator.diagnostics_.get("n_feasible", 1 if feasible else 0)),
        "l_max": l_max,
        "index": index,
        "empirical_risk": float(calibrator.diagnostics_.get("empirical_risk", 0.0)),
        "slack": float(calibrator.diagnostics_.get("slack", 0.0)),
    })
    if kind == "mixture" and feasible:
        outcomes.update({
            "threshold_low": float(calibrator.diagnostics_["threshold_low"]),
            "threshold_high": float(calibrator.diagnostics_["threshold_high"]),
            "probability_low": float(calibrator.diagnostics_["probability_low"]),
        })
    if validation is not None and selection == "validation":
        outcomes["validation_sensitivity"] = (None if validation_report is None
                                              else validation_report.get("validation_sensitivity"))
        outcomes["validation_risk"] = (None if validation_report is None
                                       else validation_report.get("validation_risk"))
        outcomes["validation_index"] = validation_index
    if selection == "utility" and validation_report is not None:
        outcomes["validation_utility"] = validation_report.get("validation_utility")
        outcomes["validation_utility_total"] = validation_report.get("validation_utility_total")
        outcomes["validation_utility_lcb"] = validation_report.get("validation_utility_lcb")
        outcomes["validation_risk"] = validation_report.get("validation_risk")
        outcomes["validation_index"] = validation_index
    return outcomes


def evaluate_at_budget(scores, onset, lengths, calibration, test, budget: float,
                       thresholds: np.ndarray, alpha: float = 0.1,
                       refractory: int = 4, escalation: int = 0,
                       lead_window_hours: float = 12.0, step_hours: float = 1.0,
                       slack: str = "bernstein",
                       min_stay_hours: float | None = None,
                       procedure: str = "blocked",
                       validation=None, selection: str = "default",
                       n_probability_points: int = 11,
                       utility_rows: np.ndarray | None = None) -> dict:
    """Calibrate on disjoint patients and evaluate only observed follow-up.

    The support bound must be specified independently of calibration/test data.
    The default allows any positive one-step stay and is conservative. A larger
    minimum defines an eligibility restriction which must hold for deployment.
    Events outside observed follow-up (or at admission) are not lead-time cases.
    """
    sequences, onset, lengths = _validate_sequences(scores, onset, lengths,
                                                    step_hours=step_hours)
    calibration, test = _validate_partitions(len(sequences), calibration, test)
    per_day = alarm_loss_matrix(sequences, lengths, thresholds, refractory=refractory,
                                escalation=escalation, step_hours=step_hours)
    return evaluate_from_losses(sequences, onset, lengths, calibration, test, per_day,
                                thresholds, budget=budget, alpha=alpha,
                                refractory=refractory, escalation=escalation,
                                lead_window_hours=lead_window_hours,
                                step_hours=step_hours, slack=slack,
                                min_stay_hours=min_stay_hours,
                                procedure=procedure, validation=validation,
                                selection=selection,
                                n_probability_points=n_probability_points,
                                utility_rows=utility_rows)


def sweep(scores, onset, lengths, calibration, test,
          budgets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
          thresholds: np.ndarray | None = None,
          **kwargs) -> list[dict]:
    """Evaluate the policy at every budget, reusing one loss matrix."""
    thresholds = np.linspace(0.05, 0.95, 200) if thresholds is None else thresholds
    sequences, onset, lengths = _validate_sequences(
        scores, onset, lengths, step_hours=kwargs.get("step_hours", 1.0))
    calibration, test = _validate_partitions(len(sequences), calibration, test)
    per_day = alarm_loss_matrix(sequences, lengths, thresholds,
                                refractory=kwargs.get("refractory", 4),
                                escalation=kwargs.get("escalation", 0),
                                step_hours=kwargs.get("step_hours", 1.0))
    return [evaluate_from_losses(sequences, onset, lengths, calibration, test, per_day,
                                 thresholds, budget=b, **kwargs)
            for b in budgets]


def pareto_front(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (alarm rate, median lead time) pairs on the non-dominated frontier."""
    points = sorted(((r["alarm_rate"], r["median_lead_time"]) for r in rows
                     if np.isfinite(r["median_lead_time"]) and np.isfinite(r["alarm_rate"])),
                    key=lambda p: (p[0], -p[1]))
    frontier = []
    best = -np.inf
    for rate, lead in points:
        if lead > best:
            frontier.append((rate, lead))
            best = lead
    if not frontier:
        return np.array([]), np.array([])
    rates, leads = zip(*frontier)
    return np.asarray(rates), np.asarray(leads)
