# Architecture

This document explains how goal-drift works in plain language. It is intended for maintainers and users who want to understand the full logic and how to integrate it correctly.

## 1) Core idea

Goal-drift prevents prompt injection by evaluating every tool action against the user's goal. It answers two questions:

1. **Is the action aligned with the user's goal?** (GoalAnchor)
2. **Does the action look harmful even if it sounds relevant?** (HarmAnchor, optional)

If the action is too far from the goal, or it resembles a known harmful pattern, the library can flag or block it.

## 2) Main classes and what they do

### DriftLevel
Simple severity enum:
- `on_task`
- `borderline`
- `off_task`

Internally, levels are ordered so the system can “escalate” to the most severe outcome.

### DriftResult
Result object returned after each check. It includes:
- `similarity`: cosine similarity between goal and action
- `level`: final classification (on_task / borderline / off_task)
- `goal_text`, `action_text`, `tool_name`, `step`
- `action_vector` (cached embedding)
- `harm_score` and `risk_score` when HarmAnchor is enabled

### GoalAnchor (required)
The main detector. It stores the user’s goal and thresholds, then classifies each action.

Key fields:
- `on_task_threshold` (default 0.40)
- `off_task_threshold` (default 0.30)
- `step_delta_threshold` (default 0.65)

Key method:
- `check(action_text, tool_name, harm_anchor=None, risk_borderline=0.0, risk_off_task=0.15)`

How `check()` works:
1. Embed the action text.
2. Compute `goal_similarity` vs. the goal embedding.
3. Classify into on_task / borderline / off_task using thresholds.
4. If HarmAnchor is present, compute `risk_score = harm_sim - goal_sim` and classify risk.
5. Final level is the more severe of goal-level and risk-level.
6. Apply step-delta escalation (borderline → off_task if the action suddenly diverges from the previous one).
7. Return a DriftResult and record history.

### HarmAnchor (optional)
Goal-agnostic harm detector. It holds a list of harm prototypes (exfiltration, deletion, unauthorized transfers, etc.).

How it works:
1. Embed each prototype once at initialization.
2. For each action, compute cosine similarity to all prototypes.
3. `harm_score = max(similarities)`

The HarmAnchor never acts alone; it is combined with GoalAnchor via:

`risk_score = harm_score - goal_similarity`

This avoids false positives when legitimate in-domain actions share vocabulary with harm prototypes.

### semantic() + build_action_description()

`@semantic` is the primary integration mechanism. It attaches a description function to tools so goal-drift can score what the tool is *actually doing*.

Rules:
- If a tool has `@semantic`, the description is generated and used for checks.
- If a tool does **not** have `@semantic`, it is **skipped** (no drift check).

This is intentional: `@semantic` is how you declare which tools are important enough to monitor.

## 3) Modes of operation

### A) Manual (no framework)
You call `build_action_description()` + `GoalAnchor.check()` yourself before executing a tool.

### B) LangChain wrapper (recommended)
`wrap_tools_with_drift_check()` wraps tool functions and enforces checks before tool execution.
- Can prompt or block.
- Doesn’t require a tool registry.

### C) LangChain callback (observability only)
`GoalDriftCallback` logs drift events but **cannot block** (LangChain swallows callback exceptions).
- Requires a `tool_registry` (tool name → function) so it can find `@semantic`.

## 4) Flow of a tool call

```text
User Goal
   |
   v
GoalAnchor (stores goal embedding + thresholds)
   |
   v
Tool call triggered
   |
   v
@semantic?  -- no --> SKIP (no drift check; tool runs)
   |
  yes
   |
   v
build_action_description(tool_fn, inputs)
   |
   v
Embed action_text
   |
   v
Goal similarity = cosine(goal, action)
   |
   v
Optional HarmAnchor:
   harm_sim = max cosine(action, harm_prototypes)
   risk_score = harm_sim - goal_sim
   risk_level = classify(risk_score)
   final_level = max(goal_level, risk_level)
   |
   v
Final DriftLevel
   |
   v
If wrapper path:
   - OFF_TASK / BORDERLINE => prompt or block
   - else allow tool
If callback path:
   - log only (cannot block)
```

## 5) Evaluation and tuning

The evaluation harness (`tests/eval_dataset.py`) runs all labeled cases in `tests/attacks/dataset.yaml` and reports:
- Goal-only metrics
- Combined goal + harm metrics (`--with-harm`)
- Harm-label confusion (if harm_label exists)

Thresholds to tune:
- Goal thresholds: in `GoalAnchor` defaults
- Risk thresholds: in `GoalAnchor.check()` parameters
- Harm label thresholds: in the eval script for reporting only

## 6) Practical guidance

- Always decorate important tools with `@semantic`.
- Reuse a single embedder instance (model load is slow).
- If you swap embedding models, re-run the eval harness and re-tune thresholds.
- Use the wrapper for enforcement, and the callback for logging.
