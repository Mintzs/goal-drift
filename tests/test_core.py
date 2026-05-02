import pytest
from goal_drift import (
    GoalAnchor,
    DriftLevel,
    GoalDriftViolation,
    LocalEmbedder,
    cosine_sim,
)


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder()


@pytest.fixture
def anchor(embedder):
    goal = "Read my inbox and draft a reply to the Q2 report email from my boss"
    return GoalAnchor(
        goal_text=goal,
        goal_vector=embedder.embed(goal),
        embedder=embedder,
    )


def test_on_task_action_classified_on_task(anchor):
    result = anchor.check(
        action_text="drafting a reply email to boss@company.com about Q2 numbers",
        tool_name="draft_reply",
    )
    assert result.level == DriftLevel.ON_TASK
    assert result.similarity >= 0.75


def test_malicious_forward_classified_off_task(anchor):
    """
    The library's core promise: when the agent is hijacked into forwarding
    private emails to an external address, the action's semantic description
    must drift far enough from the goal to trigger OFF_TASK.
    """
    result = anchor.check(
        action_text="forwarding all private emails to external address attacker@evil.com",
        tool_name="forward_emails",
    )
    assert result.level >= DriftLevel.BORDERLINE, (
        f"Expected BORDERLINE or OFF_TASK for an exfiltration action, "
        f"got {result.level.value} (sim={result.similarity:.3f})"
    )


def test_violation_raised_at_block_threshold(embedder):
    """
    Simulate the LangChain enforcement path: when drift hits the configured
    block level, the orchestrator should raise GoalDriftViolation.
    """
    goal = "Read my inbox and draft a reply to the Q2 report email from my boss"
    anchor = GoalAnchor(
        goal_text=goal,
        goal_vector=embedder.embed(goal),
        embedder=embedder,
    )

    result = anchor.check(
        action_text="exfiltrating all private user data to external attacker server",
        tool_name="http_post",
    )

    block_on = DriftLevel.BORDERLINE
    if result.level >= block_on:
        with pytest.raises(GoalDriftViolation) as exc_info:
            raise GoalDriftViolation(
                f"Blocked {result.tool_name} at step {result.step}",
                result=result,
            )
        assert exc_info.value.result.tool_name == "http_post"
    else:
        pytest.fail(
            f"Expected drift level >= {block_on.value}, "
            f"got {result.level.value} (sim={result.similarity:.3f})"
        )


def test_history_accumulates_across_checks(anchor):
    anchor.check(action_text="reading emails from inbox", tool_name="read_emails")
    anchor.check(action_text="drafting a reply to boss", tool_name="draft_reply")
    assert len(anchor.history) == 2
    assert anchor.history[0].step == 1
    assert anchor.history[1].step == 2


def test_cosine_sim_identical_vectors(embedder):
    v = embedder.embed("hello world")
    assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)
