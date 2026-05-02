from __future__ import annotations
import inspect
import json
from functools import wraps
from typing import Any, Callable

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction
except ImportError:
    raise ImportError(
        "langchain-core is required. Install with: pip install langchain-core"
    )

from .core import GoalAnchor, DriftLevel, DriftResult, GoalDriftViolation
from .embedder import BaseEmbedder, LocalEmbedder
from .harm import HarmAnchor
from .semantic import build_action_description

_ICONS = {
    DriftLevel.ON_TASK:    "✅",
    DriftLevel.BORDERLINE: "⚠️ ",
    DriftLevel.OFF_TASK:   "🚨",
}


class GoalDriftCallback(BaseCallbackHandler):
    """
    Observability-only handler that logs goal-drift events as the agent runs.

    NOTE: LangChain/LangGraph treat callbacks as advisory — exceptions raised
    inside on_tool_start are swallowed by the framework and the agent
    continues. Do NOT rely on this class to actually halt a malicious tool
    call. For real enforcement (blocking + optional human-in-the-loop), use
    `wrap_tools_with_drift_check()` instead.

    Compatible with both LangGraph (on_tool_start) and legacy AgentExecutor
    (on_agent_action).
    """

    def __init__(
        self,
        goal_text: str,
        tool_registry: dict[str, Any] | None = None,
        embedder: BaseEmbedder | None = None,
        block_on: DriftLevel = DriftLevel.OFF_TASK,
        on_flag: Any = None,
        harm_anchor: HarmAnchor | None = None,
    ):
        self._embedder = embedder or LocalEmbedder()
        self.anchor = GoalAnchor(
            goal_text=goal_text,
            goal_vector=self._embedder.embed(goal_text),
            embedder=self._embedder,
        )
        self.tool_registry = tool_registry or {}
        self.block_on = block_on
        self.on_flag = on_flag
        self.harm_anchor = harm_anchor

    # ── LangGraph / modern LangChain (create_react_agent) ────────────────────
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_name = serialized.get("name", "unknown")

        # Prefer structured inputs dict; fall back to parsing input_str as JSON
        if inputs:
            tool_input = inputs
        else:
            try:
                tool_input = json.loads(input_str)
                if not isinstance(tool_input, dict):
                    tool_input = {"input": input_str}
            except (json.JSONDecodeError, TypeError):
                tool_input = {"input": input_str}

        self._check_and_enforce(tool_name, tool_input)

    # ── Legacy AgentExecutor ──────────────────────────────────────────────────
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        tool_input = (
            action.tool_input
            if isinstance(action.tool_input, dict)
            else {"input": action.tool_input}
        )
        self._check_and_enforce(action.tool, tool_input)

    # ── Shared logic ──────────────────────────────────────────────────────────
    def _check_and_enforce(self, tool_name: str, tool_input: dict) -> None:
        tool_fn = self.tool_registry.get(tool_name)
        action_text = build_action_description(tool_fn, tool_input)
        if action_text is None:
            return
        result = self.anchor.check(action_text, tool_name, harm_anchor=self.harm_anchor)

        self._log(result)

        if result.level != DriftLevel.ON_TASK and self.on_flag:
            self.on_flag(result)

        if result.level >= self.block_on:
            extra = ""
            if result.harm_score is not None:
                extra = f" harm={result.harm_score:.3f} risk={result.risk_score:+.3f}"
            raise GoalDriftViolation(
                f"[goal-drift] 🚨 Blocked '{tool_name}' at step {result.step} — "
                f"similarity {result.similarity:.3f} to goal.{extra} Potential prompt injection.",
                result=result,
            )

    def _log(self, result: DriftResult) -> None:
        extra = ""
        if result.harm_score is not None:
            extra = f" | harm={result.harm_score:.3f} | risk={result.risk_score:+.3f}"
        print(
            f"[goal-drift] {_ICONS[result.level]} "
            f"step={result.step} | tool={result.tool_name} | "
            f"sim={result.similarity:.3f} | level={result.level.value}{extra}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement path: tool wrapping
# ─────────────────────────────────────────────────────────────────────────────


def default_confirm(result: DriftResult) -> bool:
    """
    Interactive CLI prompt asking whether a drifting tool call may proceed.

    At OFF_TASK drift, the prompt explicitly warns that this pattern is
    consistent with prompt injection so the user can judge whether to allow
    the call. Replace with a custom callable for non-CLI integrations (web
    UIs, Slack approval flows, etc.).
    """
    is_off_task = result.level == DriftLevel.OFF_TASK

    print()
    if is_off_task:
        print("=" * 70)
        print("  🚨 POSSIBLE PROMPT INJECTION — confirmation required")
        print("=" * 70)
        print()
        print("  The agent's next tool call has drifted far from your original")
        print("  goal. This pattern often indicates the agent has been hijacked")
        print("  by hidden instructions in data it just read (an email, a")
        print("  document, a web page, a tool result, etc.).")
        print()
        print("  Review the call carefully before allowing it.")
    else:
        print("=" * 70)
        print(f"  ⚠️  Goal-drift detected — confirmation required ({result.level.value})")
        print("=" * 70)

    print()
    print("  Your original goal:")
    print(f"     {result.goal_text}")
    print()
    print("  What the agent wants to do now:")
    print(f"     {result.action_text}")
    print()
    print(f"  Tool name:    {result.tool_name}")
    print(f"  Step number:  {result.step}")
    print(f"  Similarity:   {result.similarity:.3f}  (level: {result.level.value})")
    if result.harm_score is not None:
        print(f"  Harm score:   {result.harm_score:.3f}  (risk: {result.risk_score:+.3f})")
    print()
    if is_off_task:
        print("  Allowing this could leak private data, send unauthorized")
        print("  messages, or perform actions you did not request.")
        print()

    response = input("  Allow this tool to execute? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def wrap_tools_with_drift_check(
    tools: list[Any],
    anchor: GoalAnchor,
    confirm_on: DriftLevel | None = DriftLevel.OFF_TASK,
    block_on: DriftLevel | None = None,
    confirm_fn: Callable[[DriftResult], bool] = default_confirm,
    on_flag: Callable[[DriftResult], None] | None = None,
    log: bool = True,
    harm_anchor: HarmAnchor | None = None,
) -> list[Any]:
    """
    Wrap each tool's execution function with a synchronous drift check.

    Use this instead of GoalDriftCallback when you need real enforcement.
    Exceptions raised inside a tool's func propagate through the agent's
    normal tool-execution path, so a GoalDriftViolation actually halts the
    agent. Callbacks cannot do this — LangChain swallows exceptions raised
    inside callback handlers.

    By default, every OFF_TASK drift event triggers an interactive
    human-in-the-loop confirmation. The user sees a clear prompt-injection
    warning along with the tool name and the action's semantic description,
    and decides whether to allow the call. This is preferred over silent
    hard-blocking because it lets the user catch true positives without
    losing legitimate work to false positives.

    Args:
        tools: List of StructuredTool (or compatible objects exposing `.func`
            and `.name`). Each tool's `.func` is replaced in place with a
            guarded wrapper.
        anchor: GoalAnchor used for drift detection. The same anchor's
            history accumulates across tool calls.
        confirm_on: Drift level at which to ask the user via `confirm_fn`
            before letting the tool run. Default: OFF_TASK — every possible
            prompt-injection is surfaced to the user. Set to BORDERLINE for
            a more conservative posture, or None to disable confirmation.
        block_on: Optional drift level at which to unconditionally raise
            GoalDriftViolation without asking. Default: None (no automatic
            blocking; user always gets a say). Use this only in
            non-interactive setups where there's no human available, and
            it must be strictly more severe than `confirm_on`.
        confirm_fn: Callable invoked when drift hits `confirm_on`. Returns
            True to allow, False to deny. Defaults to an interactive CLI
            prompt that explains the risk; pass your own for web UIs,
            Slack approvals, etc.
        on_flag: Optional observability hook fired for any non-ON_TASK
            result.
        log: If True, prints a one-line summary of each check.
        harm_anchor: Optional HarmAnchor for catching in-domain attacks
            (actions that share goal vocabulary but resemble harm patterns
            — e.g. "issuing an unauthorized refund" against a refund-ticket
            goal). When supplied, the final level uses the more severe of
            the goal-similarity level and the relative-risk level. See
            HarmAnchor docstring for details.

    Returns:
        The same tools list, mutated so each `.func` is now guarded.
    """
    if confirm_on is not None and block_on is not None and not (block_on > confirm_on):
        raise ValueError(
            f"block_on ({block_on.value}) must be strictly more severe than "
            f"confirm_on ({confirm_on.value}); otherwise blocking fires before "
            f"the user can be asked."
        )

    for tool in tools:
        tool.func = _build_guarded(
            original_func=tool.func,
            tool_name=tool.name,
            anchor=anchor,
            block_on=block_on,
            confirm_on=confirm_on,
            confirm_fn=confirm_fn,
            on_flag=on_flag,
            log=log,
            harm_anchor=harm_anchor,
        )

    return tools


def _build_guarded(
    *,
    original_func: Callable,
    tool_name: str,
    anchor: GoalAnchor,
    block_on: DriftLevel | None,
    confirm_on: DriftLevel | None,
    confirm_fn: Callable[[DriftResult], bool],
    on_flag: Callable[[DriftResult], None] | None,
    log: bool,
    harm_anchor: HarmAnchor | None,
) -> Callable:
    @wraps(original_func)
    def guarded(*args, **kwargs):
        # Resolve runtime args to a kwargs dict so build_action_description
        # can pass them to the @semantic-supplied desc_fn.
        if args:
            try:
                bound = inspect.signature(original_func).bind(*args, **kwargs)
                tool_input = dict(bound.arguments)
            except TypeError:
                tool_input = kwargs
        else:
            tool_input = kwargs

        action_text = build_action_description(original_func, tool_input)
        if action_text is None:
            return original_func(*args, **kwargs)
        result = anchor.check(action_text, tool_name, harm_anchor=harm_anchor)

        if log:
            extra = ""
            if result.harm_score is not None:
                extra = (
                    f" | harm={result.harm_score:.3f}"
                    f" | risk={result.risk_score:+.3f}"
                )
            print(
                f"[goal-drift] {_ICONS[result.level]} step={result.step} | "
                f"tool={tool_name} | sim={result.similarity:.3f} | "
                f"level={result.level.value}{extra}"
            )

        if on_flag is not None and result.level != DriftLevel.ON_TASK:
            on_flag(result)

        # Hard-block (opt-in, only if block_on is set and reached).
        if block_on is not None and result.level >= block_on:
            extra = ""
            if result.harm_score is not None:
                extra = f" harm={result.harm_score:.3f} risk={result.risk_score:+.3f}"
            raise GoalDriftViolation(
                f"[goal-drift] 🚨 Auto-blocked '{tool_name}' at step {result.step} — "
                f"similarity {result.similarity:.3f} to goal.{extra} Potential prompt injection.",
                result=result,
            )

        # Human-in-the-loop confirmation (default at OFF_TASK).
        if confirm_on is not None and result.level >= confirm_on:
            if not confirm_fn(result):
                extra = ""
                if result.harm_score is not None:
                    extra = f" harm={result.harm_score:.3f} risk={result.risk_score:+.3f}"
                raise GoalDriftViolation(
                    f"[goal-drift] 🛑 User denied '{tool_name}' at step {result.step} — "
                    f"similarity {result.similarity:.3f} to goal.{extra} "
                    f"Treating as confirmed prompt injection.",
                    result=result,
                )

        return original_func(*args, **kwargs)

    return guarded
