from goal_drift import semantic, build_action_description


def test_build_action_description_returns_none_without_decorator():
    def tool(x: str) -> str:
        return x

    assert build_action_description(tool, {"x": "hi"}) is None


def test_build_action_description_uses_semantic_decorator():
    @semantic(lambda x: f"echoing {x}")
    def tool(x: str) -> str:
        return x

    assert build_action_description(tool, {"x": "hi"}) == "echoing hi"
