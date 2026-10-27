"""
agents/reporting/reporting_agent.py

Alert & reporting — the last box in the pipeline. Turns the full ECGState
into (a) an alert level the hospital system can route on and (b) a
plain-language clinician report.

Alert levels
------------
  CRITICAL  the beat could not be trusted at all: validation rejected it,
            or poisoning was detected and the Recovery agent quarantined /
            rejected it. No diagnosis is reported — a human must review.
  WARNING   poisoning was detected and repaired, or the decision was
            UNCERTAIN; the diagnosis is shown but flagged as coming from a
            repaired record.
  INFO      clean beat, normal diagnosis flow.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.ecg_state import ECGState
from shared.contracts import RecoveryAction, Verdict

CRITICAL, WARNING, INFO = "CRITICAL", "WARNING", "INFO"


def _alert_level(state: ECGState) -> tuple[str, str]:
    validation = state.get("validation") or {}
    if validation.get("status") == "REJECT":
        return CRITICAL, f"Signal rejected by validation: {validation.get('reason')}"

    decision = state.get("decision")
    recovery = state.get("recovery")
    if recovery is not None and recovery.action in (RecoveryAction.QUARANTINE, RecoveryAction.REJECT):
        outcome = "quarantined" if recovery.action == RecoveryAction.QUARANTINE else "rejected"
        return CRITICAL, (f"{decision.verdict.value.replace('_', ' ')} detected; record "
                          f"{outcome} — {recovery.notes}")
    if recovery is not None:
        return WARNING, f"{decision.verdict.value.replace('_', ' ')} detected and repaired ({recovery.action.value})"
    if decision is not None and decision.verdict != Verdict.CLEAN:
        return WARNING, f"decision verdict: {decision.verdict.value}"
    return INFO, "No poisoning detected"


def _evidence_rows(state: ECGState) -> list[dict[str, Any]]:
    decision = state.get("decision")
    if decision is None:
        return []
    return [{"check": e.check_name, "channel": e.channel, "score": round(e.score, 3)} for e in decision.evidence]


LOW_DIAGNOSIS_CONFIDENCE = 0.80
NOISE_FLAG_SCORE = 0.5


def _recommendations(state: ECGState, report: dict[str, Any]) -> list[str]:
    recs = []
    validation = state.get("validation") or {}
    recovery = state.get("recovery")
    if validation.get("status") == "REJECT":
        recs.append("Request a new recording — this one failed signal integrity checks.")
        return recs
    if validation.get("status") == "REPAIR":
        recs.append(f"Input was auto-repaired by validation ({', '.join(validation.get('repair_actions') or [])}).")
    if recovery is not None:
        if recovery.action in (RecoveryAction.QUARANTINE, RecoveryAction.REJECT):
            recs.append("Expert review required before this recording or its annotation is used clinically or for training.")
        elif recovery.action == RecoveryAction.RELABEL:
            recs.append(f"Annotation corrected {state.get('label')} -> {state.get('active_label')}: confirm before "
                        "using this record for training.")
        else:
            recs.append("Diagnosis is based on a repaired signal — confirm against the original recording.")
    quality = report.get("signal_quality") or {}
    if quality.get("noise_score", 0.0) >= NOISE_FLAG_SCORE:
        recs.append(f"Signal quality concern: {quality.get('noise_type')} noise detected "
                    f"(SNR {quality.get('snr_db', float('nan')):.1f} dB) — consider re-recording.")
    dx = report.get("diagnosis")
    if dx:
        if dx["confidence"] < LOW_DIAGNOSIS_CONFIDENCE:
            recs.append(f"Low diagnostic confidence ({dx['confidence']:.2f}) — clinician review advised.")
        if dx["predicted_label"] != "N":
            recs.append(f"Abnormal beat class predicted ({dx['predicted_label']}) — correlate with the full rhythm strip.")
    return recs


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        f"ECG beat {report['beat_id']} — {report['alert_level']}",
        f"  {report['alert_reason']}",
    ]
    if report["decision"]:
        d = report["decision"]
        attack = f", attack={d['attack_type']}, severity={d['severity']}" if d["attack_type"] else ""
        lines.append(f"  Integrity: verdict={d['verdict']}{attack} (confidence {d['confidence']:.2f}; "
                     f"label score {d['label_score']:.2f}, signal score {d['signal_score']:.2f})")
    if report["signal_quality"]:
        q = report["signal_quality"]
        noise = f", noise={q['noise_type']} (severity {q['noise_severity']:.2f})" if "noise_type" in q else ""
        lines.append(f"  Signal quality: SNR {q['snr_db']:.1f} dB, spectral entropy {q['spectral_entropy']:.2f}{noise}")
    if report["recovery"]:
        lines.append(f"  Recovery: {report['recovery']['action']} — {report['recovery']['notes']}")
    if report["diagnosis"]:
        dx = report["diagnosis"]
        source = " [from repaired record]" if report["repaired"] else ""
        lines.append(f"  Diagnosis: {dx['predicted_label']} (confidence {dx['confidence']:.2f}){source}")
    else:
        lines.append("  Diagnosis: withheld — manual review required")
    if report["explanation"]:
        regions = "; ".join(r["region"] for r in report["explanation"])
        lines.append(f"  Most influential regions: {regions}")
    for rec in report["recommendations"]:
        lines.append(f"  -> {rec}")
    return "\n".join(lines)


def run_reporting(state: ECGState) -> ECGState:
    """Reads the whole state, writes state["report"]."""
    level, reason = _alert_level(state)
    decision = state.get("decision")
    recovery = state.get("recovery")
    diagnosis = state.get("diagnosis") or {}
    explainability = state.get("explainability") or {}

    report: dict[str, Any] = {
        "beat_id": state.get("beat_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alert_level": level,
        "alert_reason": reason,
        "validation": {k: (state.get("validation") or {}).get(k) for k in ("status", "reason", "repair_actions")},
        "decision": None if decision is None else {
            "verdict": decision.verdict.value,
            "confidence": decision.confidence,
            "label_score": decision.label_score,
            "signal_score": decision.signal_score,
            "recommended_action": decision.recommended_action,
            "attack_type": decision.attack_type,
            "severity": decision.severity,
        },
        "evidence": _evidence_rows(state),
        "signal_quality": (state.get("signal_poisoning").artifacts.get("signal_quality")
                           if state.get("signal_poisoning") is not None else None),
        "recovery": None if recovery is None else {
            "action": recovery.action.value,
            "repair_confidence": recovery.repair_confidence,
            "notes": recovery.notes,
        },
        "repaired": recovery is not None and recovery.repaired_record is not None,
        "claimed_label": state.get("label"),
        "diagnosis_label": state.get("active_label"),
        "diagnosis": None if diagnosis.get("predicted_class") is None else {
            k: diagnosis[k] for k in ("predicted_label", "confidence", "class_probabilities")
        },
        "explanation": explainability.get("top_regions"),
    }
    report["recommendations"] = _recommendations(state, report)
    report["text"] = _render_text(report)
    state["report"] = report
    return state
