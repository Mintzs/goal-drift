from __future__ import annotations
from functools import wraps
from typing import Any, Callable


def semantic(desc_fn: Callable[..., str]):
    """
    Decorator for agent tools. Attaches a callable that produces a
    natural language description of the tool's action from its runtime arguments.

    Usage:
        @semantic(lambda receiver, body: f"sending email to {receiver} about {body}")
        def send_email(receiver, body): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.semantic_desc = desc_fn
        return wrapper
    return decorator


def build_action_description(
    tool_fn: Callable | None,
    tool_input: dict[str, Any],
) -> str | None:
    """
    Return a natural-language action description if the tool is decorated
    with @semantic. If no decorator is present, return None to indicate the
    tool should be skipped by goal-drift.
    """
    if tool_fn is None:
        return None

    if hasattr(tool_fn, "semantic_desc"):
        try:
            return tool_fn.semantic_desc(**tool_input)
        except TypeError:
            return None

    return None
