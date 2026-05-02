# goal-drift

Goal-drift is a lightweight, embedding-based detector for prompt injection in tool-using LLM agents. It compares the semantics of each tool action against the user's goal, and (optionally) a goal-agnostic harm signal, then flags or blocks actions that drift away from intent.

This repository includes:
- A small Python library (`src/goal_drift/`)
- A LangChain integration with real enforcement
- An evaluation harness and labeled dataset
- A runnable example agent

## How it works (conceptual)

Goal-drift uses two complementary signals:

1. **GoalAnchor (required)**
   - Embeds the user's goal and each tool action.
   - Computes cosine similarity to classify each action into:
     - `on_task`
     - `borderline`
     - `off_task`

2. **HarmAnchor (optional)**
   - Embeds a curated list of harmful action prototypes (exfiltration, destruction, unauthorized transfers, etc.).
   - Scores how much each action resembles known-dangerous patterns.
   - Combined with goal similarity using:

   `risk_score = harm_sim - goal_sim`

   This avoids false positives on legitimate in-domain actions while catching in-domain abuse (e.g., “unauthorized refund” for a refund task).

Both anchors use the same embedder (default: `sentence-transformers/all-MiniLM-L6-v2`) so scores are comparable.

## Architecture (files that matter)

- Core drift logic: `src/goal_drift/core.py`
  - GoalAnchor, DriftLevel, DriftResult
  - Default thresholds + step-delta escalation
- Harm signal: `src/goal_drift/harm.py`
  - HarmAnchor + prototype list
- Tool semantics: `src/goal_drift/semantic.py`
  - `@semantic` decorator used to describe tool actions
- LangChain integration: `src/goal_drift/langchain.py`
  - `wrap_tools_with_drift_check` (enforcement)
  - `GoalDriftCallback` (observability only)
- Evaluation harness: `tests/eval_dataset.py`
- Labeled dataset: `tests/attacks/dataset.yaml`
- Example agent: `examples/basic_langchain_agent.py`

## Install

```bash
# From PyPI (once published):
pip install goal-drift
# With LangChain support:
pip install "goal-drift[langchain]"

# From source (this repo):
pip install -e .
```

Optional extras for dev:

```bash
pip install -e ".[dev]"
```

Note: the eval harness imports `yaml`; install `PyYAML` if you run it locally.

## Recommended setup: `@semantic` tool descriptions

The main integration surface is the `@semantic` decorator. It attaches a callable that turns real tool inputs into a natural-language semantic action description. That description is what GoalAnchor (and HarmAnchor) actually score.

- If a tool is missing `@semantic`, goal-drift **skips that tool call entirely** (no similarity check is performed).
- `build_action_description()` reads the decorator and formats a stable, semantic action string.
- The LangChain wrapper calls `build_action_description()` for you automatically, so you only need to annotate tools.

Minimal example (manual agent loop):

```python
from goal_drift import GoalAnchor, LocalEmbedder, semantic, build_action_description

@semantic(lambda recipient, subject, body: f"drafting a reply email to {recipient} with subject: {subject}")
def draft_reply(recipient: str, subject: str, body: str) -> str:
    ...

goal = "Read my inbox and draft a reply to the Q2 report email from my boss"
embedder = LocalEmbedder()

anchor = GoalAnchor(goal_text=goal, goal_vector=embedder.embed(goal), embedder=embedder)

tool_input = {"recipient": "boss@company.com", "subject": "Re: Q2 Report", "body": "..."}
action_text = build_action_description(draft_reply, tool_input)

result = anchor.check(action_text, tool_name="draft_reply")
print(result.level.value, result.similarity)
```

### Flow diagram (single tool call)

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

## Quick start (goal + harm)

```python
from goal_drift import GoalAnchor, HarmAnchor, LocalEmbedder, semantic, build_action_description

@semantic(lambda account_id, amount: f"issuing a refund of {amount} for account {account_id}")
def issue_refund(account_id: str, amount: str) -> None:
    ...

goal = "Answer support ticket #1029 about a failed payment refund"
embedder = LocalEmbedder()

goal_anchor = GoalAnchor(
    goal_text=goal,
    goal_vector=embedder.embed(goal),
    embedder=embedder,
)
harm_anchor = HarmAnchor(embedder=embedder)

tool_input = {"account_id": "cust_1029", "amount": "$5000"}
action_text = build_action_description(issue_refund, tool_input)

result = goal_anchor.check(
    action_text=action_text,
    tool_name="issue_refund",
    harm_anchor=harm_anchor,
)

print(result.level.value, result.similarity, result.harm_score, result.risk_score)
```

## LangChain integration (enforcement)

As of now, only LangChain is supported out of the box. There are two integration paths:

- **Wrapper (recommended):** `wrap_tools_with_drift_check` wraps tool functions and can block or prompt before execution.
- **Callback (observability only):** `GoalDriftCallback` logs drift events but cannot block (LangChain swallows exceptions raised in callbacks).

```python
from langchain_core.tools import StructuredTool
from goal_drift import GoalAnchor, HarmAnchor, LocalEmbedder, semantic
from goal_drift.langchain import wrap_tools_with_drift_check

@semantic(lambda folder: f"reading emails from the {folder} folder")
def read_emails(folder: str) -> str:
    ...

@semantic(lambda recipient, subject, body: f"drafting a reply email to {recipient} with subject: {subject}")
def draft_reply(recipient: str, subject: str, body: str) -> str:
    ...

embedder = LocalEmbedder()
goal = "Read my inbox and draft a reply to the Q2 report email from my boss"

anchor = GoalAnchor(goal_text=goal, goal_vector=embedder.embed(goal), embedder=embedder)
harm_anchor = HarmAnchor(embedder=embedder)

tools = [StructuredTool.from_function(read_emails), StructuredTool.from_function(draft_reply)]
guarded = wrap_tools_with_drift_check(
    tools,
    anchor,
    confirm_on=DriftLevel.OFF_TASK,   # prompt on off_task
    harm_anchor=harm_anchor,          # optional but recommended
)
```

### Callback usage (telemetry only)

If you use `GoalDriftCallback`, you must supply a `tool_registry` mapping tool name → function so the callback can access the `@semantic` description. Without this registry, tools are skipped because goal-drift cannot build an action description.

```python
from goal_drift import semantic
from goal_drift.langchain import GoalDriftCallback

@semantic(lambda folder: f"reading emails from the {folder} folder")
def read_emails(folder: str) -> str:
    ...

tool_registry = {
    "read_emails": read_emails,
}

callback = GoalDriftCallback(
    goal_text="Read my inbox and draft a reply",
    tool_registry=tool_registry,
)
```

### When do you need a tool registry?

- **Wrapper path:** no registry needed — the wrapper already has the tool function.
- **Callback path:** registry required — the callback only sees a tool name.
- **Manual use (no framework):** no registry needed — you call `build_action_description()` on the tool function yourself.

See `examples/basic_langchain_agent.py` for a full runnable demo using OpenRouter and LangGraph.

## Evaluation and performance

Run the eval harness from the project root:

```bash
python tests/eval_dataset.py
python tests/eval_dataset.py --with-harm
```

The dataset lives in `tests/attacks/dataset.yaml` and now includes both:
- `expected_level` (goal-alignment: on_task / borderline / off_task)
- `harm_label` (goal-agnostic: benign / suspicious / harmful)

### Latest local eval (MiniLM, dataset.yaml)

Goal-only detector:
- Accuracy: **69.57%** (32/46)
- OFF_TASK recall: **76.47%**

Goal + harm combined:
- Accuracy: **78.26%** (36/46)
- OFF_TASK recall: **100%** (17/17)
- On-task recall: **100%** (13/13)
- Harm-label accuracy: **82.61%** (38/46)

These numbers depend on the embedder and dataset. Re-run after changing thresholds, prototypes, or labels.

## Tuning and thresholds

Goal thresholds live on `GoalAnchor`:
- `on_task_threshold` (default 0.40)
- `off_task_threshold` (default 0.30)
- `step_delta_threshold` (default 0.65)

Harm integration uses the **relative** score:

`risk_score = harm_sim - goal_sim`

The default risk thresholds are in `GoalAnchor.check()`:
- `risk_borderline = 0.0`
- `risk_off_task = 0.15`

The eval harness also prints suggested goal thresholds and uses fixed harm-label thresholds (0.35 / 0.45) for reporting.

## Best practices

- Treat `@semantic` as required for production use. Missing decorators skip that tool from drift checks.
- Reuse a single `LocalEmbedder` instance; model load is slow on first use.
- If you change embedder model, re-calibrate thresholds with the eval harness.

## Limitations

- The harm anchor is a prototype-based heuristic, not a classifier.
- The dataset is small and subjective; label tuning is expected.
- Only LangChain has built-in enforcement helpers; other frameworks should call `GoalAnchor.check()` directly.

## Deep-dive architecture

For a plain-language, class-by-class walkthrough (and how the modes fit together), see `docs/ARCHITECTURE.md`.

## Packaging and publishing

This project uses `hatchling` (see `pyproject.toml`). To build and upload to PyPI:

```bash
pip install -e ".[dev]"
python -m build
python -m twine upload dist/*
```

To verify the built artifacts locally:

```bash
python -m twine check dist/*
```
