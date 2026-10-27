# Decision agent

Fuses the label-poisoning agent's calibrated `label_score` with the
signal-poisoning agent's four raw check scores into one verdict:
`clean | label_poisoned | signal_poisoned | both_poisoned | uncertain`.

## Files

```
agents/decision/
├── __init__.py
├── decision_agent.py     # DecisionAgent, DecisionAgentConfig
├── fusion.py              # signal-side score fusion (weighted mean / noisy-OR)
├── config/
│   └── thresholds.yaml    # label/signal thresholds + fusion weights
├── evaluation/
│   └── evaluate_fusion.py # offline metrics + threshold sweep against a labeled attack-sim set
└── tests/
    ├── conftest.py
    ├── test_fusion.py
    └── test_decision_agent.py

shared/
└── contracts.py            # Evidence / Verdict / DecisionRecord — MERGE into your existing contracts.py
```

## Install

```bash
pip install pydantic>=2 pyyaml scikit-learn numpy
```

(`scikit-learn` and `numpy` are only needed for `evaluation/evaluate_fusion.py`, not for the agent itself.)

## Usage

```python
from agents.decision.decision_agent import DecisionAgent
from shared.contracts import Evidence

agent = DecisionAgent()  # loads agents/decision/config/thresholds.yaml

signal_evidence = [
    Evidence(check_name="reconstruction_error", channel="signal", score=0.12),
    Evidence(check_name="feature_space_distance", channel="signal", score=0.08),
    Evidence(check_name="prediction_fragility", channel="signal", score=0.05),
    Evidence(check_name="gradient_sensitivity", channel="signal", score=0.10),
]

decision = agent.run(record_id="rec_0001", label_score=0.03, signal_evidence=signal_evidence)
print(decision.verdict, decision.recommended_action, decision.confidence)
```

## Run the tests

```bash
pip install pytest
pytest agents/decision/tests -v
```

## Calibrate thresholds against real data

Build a JSONL file of records with known ground truth (see the docstring
at the top of `evaluation/evaluate_fusion.py` for the exact format), then:

```bash
python -m agents.decision.evaluation.evaluate_fusion --data attack_sim.jsonl
python -m agents.decision.evaluation.evaluate_fusion --data attack_sim.jsonl --sweep-thresholds
```

## Integrating into your repo

1. If you already have a top-level `contracts.py` (per your existing repo
   layout), merge `Evidence`, `Verdict`, `DecisionRecord`, and
   `AgentEnvelope` from `shared/contracts.py` into it instead of keeping
   two copies. Every other agent must import these from one place.
2. Drop `agents/decision/` in next to your `agents/validation/`,
   `agents/signal_poisoning/`, etc.
3. The orchestrator calls `DecisionAgent().run(record_id, label_score,
   signal_evidence)` after both detection agents finish — see Section 9
   of the build-plan document for the full `Pipeline` wiring.
4. Before wiring in real detectors: `label_score` must already be a
   single calibrated float in `[0,1]` from the label-poisoning agent's
   "Score fusion" step. `signal_evidence` must contain exactly the four
   checks named in `thresholds.yaml`'s `signal_check_weights` — the
   agent raises a clear `ValueError` if one is missing, rather than
   silently fusing a partial set.
