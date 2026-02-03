"""Tests for resource extractors."""

from pathlib import Path

from tracemem_core.extractors import DefaultResourceExtractor, _canonicalize_file_uri


class TestCanonicalizeFileUri:
    """Test _canonicalize_file_uri helper."""

    def test_http_url_passes_through(self) -> None:
        """Non-file URIs pass through unchanged."""
        assert _canonicalize_file_uri("https://example.com/api", root=None) == "https://example.com/api"

    def test_custom_scheme_passes_through(self) -> None:
        """Custom URI schemes pass through unchanged."""
        assert _canonicalize_file_uri("ticker://AAPL", root=None) == "ticker://AAPL"

    def test_absolute_path_without_root(self, tmp_path: Path) -> None:
        """Without root, absolute file URIs stay absolute."""
        test_file = tmp_path / "file.py"
        test_file.touch()
        result = _canonicalize_file_uri(f"file://{test_file}", root=None)
        assert result == f"file://{test_file.resolve()}"

    def test_absolute_path_with_root_makes_relative(self, tmp_path: Path) -> None:
        """With root, file URIs under root become relative."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "auth.py"
        test_file.touch()

        result = _canonicalize_file_uri(f"file://{test_file}", root=tmp_path)
        assert result == "file://src/auth.py"

    def test_path_outside_root_stays_absolute(self, tmp_path: Path) -> None:
        """File URIs outside root stay absolute."""
        test_file = tmp_path / "outside.py"
        test_file.touch()
        other_root = tmp_path / "project"
        other_root.mkdir()

        result = _canonicalize_file_uri(f"file://{test_file}", root=other_root)
        assert result == f"file://{test_file.resolve()}"


class TestDefaultResourceExtractor:
    """Test DefaultResourceExtractor."""

    def test_extract_file_path(self, tmp_path: Path) -> None:
        """Extract URI from path argument."""
        test_file = tmp_path / "file.py"
        test_file.touch()
        extractor = DefaultResourceExtractor()

        result = extractor.extract("read_file", {"path": str(test_file)})

        assert result == f"file://{test_file.resolve()}"

    def test_extract_file_path_variant(self, tmp_path: Path) -> None:
        """Extract URI from file_path argument."""
        test_file = tmp_path / "file.py"
        test_file.touch()
        extractor = DefaultResourceExtractor()

        result = extractor.extract("read", {"file_path": str(test_file)})

        assert result == f"file://{test_file.resolve()}"

    def test_extract_local_mode_returns_relative(self, tmp_path: Path) -> None:
        """Extractor in local mode makes file paths relative."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "auth.py"
        test_file.touch()

        extractor = DefaultResourceExtractor(mode="local", home=tmp_path / ".tracemem")
        result = extractor.extract("read", {"file_path": str(test_file)})
        assert result == "file://src/auth.py"

    def test_extract_global_mode_returns_absolute(self, tmp_path: Path) -> None:
        """Extractor in global mode uses absolute paths."""
        test_file = tmp_path / "auth.py"
        test_file.touch()

        extractor = DefaultResourceExtractor(mode="global")
        result = extractor.extract("read", {"file_path": str(test_file)})
        assert result == f"file://{test_file.resolve()}"

    def test_extract_default_mode_is_global(self, tmp_path: Path) -> None:
        """Default mode is global (absolute paths)."""
        test_file = tmp_path / "auth.py"
        test_file.touch()

        extractor = DefaultResourceExtractor()
        result = extractor.extract("read", {"file_path": str(test_file)})
        assert result == f"file://{test_file.resolve()}"

    def test_extract_url_passes_through(self) -> None:
        """Non-file URIs pass through unchanged."""
        extractor = DefaultResourceExtractor(mode="local", home=Path("/some/project/.tracemem"))
        result = extractor.extract("fetch", {"url": "https://example.com/api"})
        assert result == "https://example.com/api"

    def test_extract_uri(self) -> None:
        """Extract URL from uri argument."""
        extractor = DefaultResourceExtractor()

        result = extractor.extract("fetch", {"uri": "https://example.com/api"})

        assert result == "https://example.com/api"

    def test_extract_endpoint(self) -> None:
        """Extract URL from endpoint argument."""
        extractor = DefaultResourceExtractor()

        result = extractor.extract("api_call", {"endpoint": "https://api.example.com"})

        assert result == "https://api.example.com"

    def test_extract_no_match(self) -> None:
        """Return None when no known arguments found."""
        extractor = DefaultResourceExtractor()

        result = extractor.extract("some_tool", {"data": "value", "count": 5})

        assert result is None

    def test_extract_empty_value(self) -> None:
        """Return None when argument is empty."""
        extractor = DefaultResourceExtractor()

        result = extractor.extract("read", {"path": ""})

        assert result is None

    def test_extract_none_value(self) -> None:
        """Return None when argument is None."""
        extractor = DefaultResourceExtractor()

        result = extractor.extract("read", {"path": None})

        assert result is None

    def test_extract_file_takes_precedence(self, tmp_path: Path) -> None:
        """File args should be checked before URL args."""
        test_file = tmp_path / "file.py"
        test_file.touch()
        extractor = DefaultResourceExtractor()

        result = extractor.extract(
            "some_tool", {"path": str(test_file), "url": "https://example.com"}
        )

        assert result == f"file://{test_file.resolve()}"

    def test_extract_non_string_value(self) -> None:
        """Return None for non-string values."""
        extractor = DefaultResourceExtractor()

        result = extractor.extract("read", {"path": 123})

        assert result is None
