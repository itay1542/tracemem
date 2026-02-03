"""Unit tests for reranker registry."""

from unittest.mock import MagicMock

import pytest
from lancedb.rerankers import LinearCombinationReranker, RRFReranker

from tracemem_core.storage.vector.rerankers import get_reranker


class TestGetReranker:
    """Tests for get_reranker() resolution."""

    def test_rrf_string_returns_rrf_reranker(self):
        """String "rrf" resolves to an RRFReranker instance."""
        result = get_reranker("rrf")
        assert isinstance(result, RRFReranker)

    def test_linear_string_returns_linear_combination_reranker(self):
        """String "linear" resolves to a LinearCombinationReranker instance."""
        result = get_reranker("linear")
        assert isinstance(result, LinearCombinationReranker)

    def test_unknown_string_raises_value_error(self):
        """Unknown reranker string raises ValueError with available options."""
        with pytest.raises(ValueError, match="Unknown reranker 'unknown'"):
            get_reranker("unknown")

    def test_custom_instance_passes_through(self):
        """A custom reranker instance is returned as-is."""
        custom = MagicMock()
        result = get_reranker(custom)
        assert result is custom
