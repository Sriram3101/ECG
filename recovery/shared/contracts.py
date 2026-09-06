"""
shared/contracts.py

Cross-agent schemas. This file contains the subset needed by the Recovery
agent (ECGRecord, RecoveryAction, RecoveryResult), plus the upstream
schemas it depends on (AgentEnvelope, Evidence, Verdict, DecisionRecord).

If you already have a contracts.py in your repo (e.g. from the Decision
agent package), MERGE these classes into it rather than having two
competing definitions — every agent must import from one place. The
AgentEnvelope / Evidence / Verdict / DecisionRecord classes below are
byte-for-byte the same ones shipped with the Decision agent package, so
merging is just a union of the two files.

Requires: pydantic>=2
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator


class AgentEnvelope(BaseModel):
    """Common metadata every agent input/output carries."""

    record_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    agent_version: str
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}


class ECGRecord(BaseModel):
    """Canonical signal object passed between every agent."""

    record_id: str
    signal: list[float]  # length 360, single-lead beat window
    label: Optional[int] = None
    patient_id: Optional[str] = None
    source: Literal["raw", "repaired", "quarantined"] = "raw"

    @field_validator("signal")
    @classmethod
    def check_shape(cls, v):
        arr = np.asarray(v, dtype=float)
        if arr.shape != (360,):
            raise ValueError(f"expected shape (360,), got {arr.shape}")
        if np.isnan(arr).any():
            raise ValueError("signal contains NaN")
        return v


class Evidence(BaseModel):
    """One detector's opinion. `score` MUST be normalized to [0, 1],
    higher = more suspicious. `metadata` is the side-channel for
    artifacts the Recovery agent needs later (e.g. the autoencoder
    reconstruction array), so it doesn't have to be recomputed."""

    check_name: str
    channel: Literal["label", "signal"]
    score: float = Field(ge=0.0, le=1.0)
    raw_value: Optional[float] = None
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class Verdict(str, Enum):
    CLEAN = "clean"
    LABEL_POISONED = "label_poisoned"
    SIGNAL_POISONED = "signal_poisoned"
    BOTH_POISONED = "both_poisoned"
    UNCERTAIN = "uncertain"


class DecisionRecord(AgentEnvelope):
    verdict: Verdict
    confidence: float
    label_score: float
    signal_score: float
    evidence: list[Evidence]
    recommended_action: Literal["proceed", "repair", "quarantine", "reject"]


class RecoveryAction(str, Enum):
    RELABEL = "relabel"
    RECONSTRUCT_SIGNAL = "reconstruct_signal"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    NONE = "none"


class RecoveryResult(AgentEnvelope):
    action: RecoveryAction
    repaired_record: Optional[ECGRecord] = None
    repair_confidence: Optional[float] = None
    notes: str = ""
