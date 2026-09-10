"""Basic tests for query scaffolding."""

from reifyscene.query.parser import PromptParser


def test_prompt_parser_returns_mode() -> None:
    parsed = PromptParser().parse("show me the couch view")
    assert parsed["mode"] == "faithful"
