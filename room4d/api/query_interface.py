"""High-level query-time interface."""

class QueryInterface:
    """Run prompt parsing, retrieval, planning, and rendering."""

    def answer(self, prompt: str) -> dict:
        """Return placeholder query outputs."""
        return {"prompt": prompt}
