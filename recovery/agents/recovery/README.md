# Recovery agent

Repairs or quarantines a record after the Decision agent has flagged it.
Only ever invoked when `decision.verdict != CLEAN` — the orchestrator
should route CLEAN records straight to Diagnosis without calling this
agent at all (`RecoveryAgent.run()` raises if you call it on CLEAN).

| Verdict | Action |
|---|---|
| `label_poisoned` | Re-label using the label-poisoning agent's confident-learning OOF consensus, if confident enough; otherwise quarantine |
| `signal_poisoned` | Substitute the autoencoder's reconstruction, if the reconstruction is itself trustworthy enough; otherwise quarantine |
| `both_poisoned` | Always quarantine — compounding risk, no automatic repair |
| `uncertain` | Always quarantine — routed to manual review |

## A design note worth reading before wiring this in

`signal_repair.py`'s trust gate is **`reconstruction_confidence`**, not
"how different is the reconstruction from the original poisoned
signal." A real repair for a heavily-poisoned signal is *supposed* to
look very different from the input — that's the point. Gating on that
difference would quarantine almost every genuine attack and only
"repair" signals that barely needed it (see
`test_large_difference_from_original_does_not_block_repair` in the
tests, which pins this down as a regression test).

`reconstruction_confidence` should represent how in-distribution /
typical the reconstruction is relative to the autoencoder's own
training distribution of clean beats — e.g. a calibrated percentile of
its reconstruction-loss distribution on held-out clean data, or a
dropout/ensemble uncertainty estimate. If your signal-poisoning agent
doesn't produce this yet, that's a gap in that agent, not something to
approximate here from the original-vs-reconstruction distance.

## Files

```
agents/recovery/
├── __init__.py
├── recovery_agent.py          # RecoveryAgent, RecoveryAgentConfig, artifact dispatch
├── strategies/
│   ├── label_repair.py         # confident-learning OOF consensus relabel
│   ├── signal_repair.py        # autoencoder-reconstruction substitution
│   └── quarantine.py           # quarantine() / reject()
├── config/
│   └── repair_thresholds.yaml  # min_label_confidence, min_signal_confidence
├── evaluation/
│   └── evaluate_recovery.py    # relabel accuracy + signal-repair MAE improvement vs. ground truth
└── tests/
    ├── conftest.py
    ├── test_label_repair.py
    ├── test_signal_repair.py
    ├── test_quarantine.py
    └── test_recovery_agent.py

shared/
└── contracts.py                 # ECGRecord, RecoveryAction, RecoveryResult, + upstream schemas
                                  # MERGE into your existing contracts.py
```

## Install

```bash
pip install pydantic>=2 pyyaml numpy
```

(`numpy` is used for validation math; nothing here needs `scikit-learn`.)

## Usage

```python
from agents.recovery.recovery_agent import RecoveryAgent
from shared.contracts import ECGRecord

agent = RecoveryAgent()  # loads agents/recovery/config/repair_thresholds.yaml

# `decision` is a DecisionRecord from the Decision agent (verdict != CLEAN)
# `record` is the flagged ECGRecord

result = agent.run(
    decision, record,
    oof_probabilities=[0.02, 0.90, 0.05, 0.02, 0.01],           # for label_poisoned
    autoencoder_reconstruction=[...],                             # for signal_poisoned
    reconstruction_confidence=0.85,                               # for signal_poisoned
)
print(result.action, result.repair_confidence, result.notes)
```

Or let it pull `autoencoder_reconstruction` / `reconstruction_confidence`
/ `oof_probabilities` automatically from `decision.evidence` metadata —
see `RecoveryAgent.extract_artifacts()` and its docstring for the exact
keys the detection agents should populate.

## Run the tests

```bash
pip install pytest
pytest agents/recovery/tests -v
```

## Evaluate against ground truth

Build a JSONL file with known clean labels/signals underneath simulated
attacks (see the docstring at the top of
`evaluation/evaluate_recovery.py` for the exact format), then:

```bash
python -m agents.recovery.evaluation.evaluate_recovery --data attack_sim.jsonl
```

This reports the action distribution, relabel accuracy against ground
truth, and mean-absolute-error before vs. after signal repair — the
thing that actually tells you whether repair is helping or hurting,
not just that it ran.

## Integrating into your repo

1. Merge `ECGRecord`, `RecoveryAction`, `RecoveryResult` (and the
   upstream `AgentEnvelope` / `Evidence` / `Verdict` / `DecisionRecord`
   if not already there from the Decision agent) into your existing
   `contracts.py`.
2. Drop `agents/recovery/` in next to `agents/decision/`.
3. The orchestrator calls this **only** when
   `decision.recommended_action != "proceed"`. If the result's
   `.action` is `QUARANTINE` or `REJECT`, stop the pipeline there —
   don't forward to Diagnosis. Otherwise forward `result.repaired_record`
   to Diagnosis instead of the original record.
4. Decide now, with whoever owns the label-poisoning and
   signal-poisoning agents, how `oof_probabilities` and
   `(autoencoder_reconstruction, reconstruction_confidence)` will reach
   this agent — as orchestrator-level artifacts or as `Evidence.metadata`
   / `Evidence.confidence`. Both paths are supported here; pick one and
   be consistent so it isn't a silent fallback nobody remembers exists.
