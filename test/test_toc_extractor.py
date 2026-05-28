"""Tests for TOC extractor functions."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from core.utils.toc_extractor import (
    extract_pages,
    llm_detect_toc,
    find_toc_pages,
    extract_toc_content,
    detect_page_index,
    _invoke_with_retry,
)
from core.schemas.toc import TOCDetection, TOCContent, PageIndexDetection


class TestExtractPages:
    def test_empty_doc(self):
        doc = Mock()
        doc.pages = {}
        assert extract_pages(doc) == []

    def test_extracts_text_per_page(self):
        doc = Mock()
        doc.pages = {1: "page1", 2: "page2"}

        item1 = Mock()
        item1.text = "Hello"
        item1.prov = [Mock(page_no=1)]

        item2 = Mock()
        item2.text = "World"
        item2.prov = [Mock(page_no=2)]

        doc.iterate_items.return_value = [(item1, 0), (item2, 0)]

        result = extract_pages(doc)
        assert len(result) == 2
        assert "Hello" in result[0]
        assert "World" in result[1]


class TestInvokeWithRetry:
    def test_success_first_try(self):
        mock_llm = Mock()
        mock_llm.invoke.return_value = "success"
        result = _invoke_with_retry(mock_llm, "test prompt")
        assert result == "success"

    def test_fallback_after_retries(self):
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("fail")
        result = _invoke_with_retry(mock_llm, "test prompt", max_retries=2)
        assert result is None


class TestLlmDetectToc:
    def test_short_text_returns_false(self):
        result = llm_detect_toc("hi", Mock())
        assert result is False

    def test_empty_text_returns_false(self):
        result = llm_detect_toc("", Mock())
        assert result is False

    def test_valid_toc_detection(self):
        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = TOCDetection(is_toc=True)
        mock_llm.with_structured_output.return_value = mock_structured

        result = llm_detect_toc("Table of Contents\nChapter 1 ... 10\nChapter 2 ... 20", mock_llm)
        assert result is True

    def test_returns_false_on_error(self):
        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_structured

        result = llm_detect_toc("Table of Contents\nChapter 1 ... 10", mock_llm)
        assert result is False


class TestFindTocPages:
    def test_empty_pages(self):
        result = find_toc_pages([], Mock())
        assert result == []

    def test_no_toc_found(self):
        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = TOCDetection(is_toc=False)
        mock_llm.with_structured_output.return_value = mock_structured

        result = find_toc_pages(["Some text without TOC"], mock_llm)
        assert result == []

    def test_all_toc_pages(self):
        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = TOCDetection(is_toc=True)
        mock_llm.with_structured_output.return_value = mock_structured

        pages = ["TOC page 1\nwith enough text", "TOC page 2\nwith enough text"]
        result = find_toc_pages(pages, mock_llm)
        assert result == [0, 1]


class TestExtractTocContent:
    def test_empty_indices(self):
        result = extract_toc_content(["page1"], Mock(), [])
        assert result == ""

    def test_extracts_content(self):
        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = TOCContent(
            toc_text="1. Chapter 1\n2. Chapter 2",
            chapters=[],
        )
        mock_llm.with_structured_output.return_value = mock_structured

        result = extract_toc_content(
            ["1. Chapter 1\n2. Chapter 2"],
            mock_llm,
            [0],
        )
        assert "Chapter 1" in result


class TestDetectPageIndex:
    def test_empty_text(self):
        result = detect_page_index("", Mock())
        assert result is False

    def test_detects_page_numbers(self):
        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = PageIndexDetection(
            page_index_given_in_toc=True
        )
        mock_llm.with_structured_output.return_value = mock_structured

        result = detect_page_index("1. Chapter 1 ... 10", mock_llm)
        assert result is True
