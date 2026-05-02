from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .embedder import BaseEmbedder
    from .harm import HarmAnchor


# --- Severity ordering for threshold comparisons ---
_SEVERITY: dict[str, int] = {
    "on_task":    0,
    "borderline": 1,
    "off_task":   2,
}


class DriftLevel(Enum):
    """
    Three-level classification of an agent action against the user's goal.

    ON_TASK     — action is semantically aligned with the goal; let it run.
    BORDERLINE  — tangential or fuzzy; log/observe, optionally ask the user.
    OFF_TASK    — strong drift signal consistent with prompt injection; the
                  user should review before this tool executes.
    """

    ON_TASK    = "on_task"
    BORDERLINE = "borderline"
    OFF_TASK   = "off_task"

    def __ge__(self, other: DriftLevel) -> bool:
        return _SEVERITY[self.value] >= _SEVERITY[other.value]

    def __gt__(self, other: DriftLevel) -> bool:
        return _SEVERITY[self.value] > _SEVERITY[other.value]


@dataclass
class DriftResult:
    similarity: float
    level: DriftLevel
    goal_text: str
    action_text: str
    action_vector: np.ndarray   # stored to avoid re-embedding in delta checks
    tool_name: str
    step: int
    # Populated only when GoalAnchor.check() was called with a harm_anchor.
    harm_score: float | None = None
    risk_score: float | None = None  # harm_score - similarity

    def __repr__(self) -> str:
        return (
            f"DriftResult(step={self.step}, tool={self.tool_name!r}, "
            f"similarity={self.similarity:.3f}, level={self.level.value!r})"
        )


class GoalDriftViolation(Exception):
    def __init__(self, message: str, result: DriftResult):
        super().__init__(message)
        self.result = result


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Assumes both vectors are already L2-normalized (normalize_embeddings=True)
    return float(np.dot(a, b))


@dataclass
class GoalAnchor:
    goal_text: str
    goal_vector: np.ndarray
    embedder: "BaseEmbedder"
    # Defaults derived empirically from tests/attacks/dataset.yaml on
    # sentence-transformers/all-MiniLM-L6-v2. Recalibrate against your own
    # labeled dataset if you swap embedders.
    on_task_threshold:  float = 0.40
    off_task_threshold: float = 0.30
    # step_delta_threshold: how much a single step can jump before escalating level
    step_delta_threshold: float = 0.65

    step_count: int              = field(default=0, init=False)
    history: list[DriftResult]   = field(default_factory=list, init=False)

    def _classify(self, sim: float) -> DriftLevel:
        if sim >= self.on_task_threshold:
            return DriftLevel.ON_TASK
        elif sim >= self.off_task_threshold:
            return DriftLevel.BORDERLINE
        return DriftLevel.OFF_TASK

    def check(
        self,
        action_text: str,
        tool_name: str,
        harm_anchor: "HarmAnchor | None" = None,
        risk_borderline: float = 0.0,
        risk_off_task: float = 0.15,
    ) -> DriftResult:
        """
        Classify an action against the goal, optionally combining with a
        goal-agnostic harm signal.

        When `harm_anchor` is None (default), behavior is unchanged: pure
        goal-similarity classification using the GoalAnchor's thresholds.

        When `harm_anchor` is supplied, we additionally compute:
            risk_score = harm_anchor.score(action) - goal_similarity
        and classify the risk into ON_TASK / BORDERLINE / OFF_TASK using
        `risk_borderline` and `risk_off_task`. The final level is the more
        severe of (goal_level, risk_level). This catches in-domain attacks
        — actions that share goal vocabulary but also resemble known harm
        patterns. See harm.py for the rationale.
        """
        self.step_count += 1

        action_vector = self.embedder.embed(action_text)
        sim = cosine_sim(self.goal_vector, action_vector)
        goal_level = self._classify(sim)

        harm_score: float | None = None
        risk_score: float | None = None
        if harm_anchor is not None:
            harm_score = harm_anchor.score(action_vector)
            risk_score = harm_score - sim
            risk_level = _classify_risk(risk_score, risk_borderline, risk_off_task)
            level = _max_severity(goal_level, risk_level)
        else:
            level = goal_level

        # Step delta: a sudden behavioral jump between consecutive steps
        # escalates BORDERLINE → OFF_TASK even if absolute sim looks okay.
        if self.history and level == DriftLevel.BORDERLINE:
            step_delta = 1.0 - cosine_sim(self.history[-1].action_vector, action_vector)
            if step_delta > self.step_delta_threshold:
                level = DriftLevel.OFF_TASK

        result = DriftResult(
            similarity=sim,
            level=level,
            goal_text=self.goal_text,
            action_text=action_text,
            action_vector=action_vector,
            tool_name=tool_name,
            step=self.step_count,
            harm_score=harm_score,
            risk_score=risk_score,
        )
        self.history.append(result)
        return result


def _classify_risk(risk_score: float, borderline_t: float, off_task_t: float) -> DriftLevel:
    if risk_score >= off_task_t:
        return DriftLevel.OFF_TASK
    if risk_score >= borderline_t:
        return DriftLevel.BORDERLINE
    return DriftLevel.ON_TASK


def _max_severity(a: DriftLevel, b: DriftLevel) -> DriftLevel:
    return a if _SEVERITY[a.value] >= _SEVERITY[b.value] else b
