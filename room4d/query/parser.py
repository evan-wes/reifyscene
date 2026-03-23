"""Prompt parsing helpers."""

class PromptParser:
    """Convert a natural-language prompt into structured query constraints."""

    def parse(self, prompt: str) -> dict:
        """Return a placeholder parsed prompt."""
        return {
            "subject": None,
            "time_constraint": None,
            "view_constraint": None,
            "mode": "faithful",
            "raw_prompt": prompt,
        }
