"""Tests for all Pydantic schemas."""
import pytest
from pydantic import ValidationError

from core.schemas.expertise import ExpertiseDetection
from core.schemas.chapter import ChapterIdentification
from core.schemas.mindmap import MindmapNode, MindmapResponse
from core.schemas.question import (
    MCQOption,
    MCQQuestion,
    MCQResponse,
    EssayQuestion,
    EssayResponse,
)
from core.schemas.chat import Message, ChatPayload
from core.schemas.toc import (
    TOCDetection,
    TOCChapter,
    TOCContent,
    PageIndexDetection,
)


# ------------------ ExpertiseDetection ------------------
class TestExpertiseDetection:
    def test_valid(self):
        result = ExpertiseDetection(
            domain="Software Engineering",
            sub_fields=["code quality", "refactoring"],
            expertise_prompt="ahli Software Engineering",
            book_type="textbook",
        )
        assert result.domain == "Software Engineering"
        assert len(result.sub_fields) == 2

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ExpertiseDetection()


# ------------------ ChapterIdentification ------------------
class TestChapterIdentification:
    def test_valid(self):
        result = ChapterIdentification(chapters=["Bab 1", "Bab 2", "Bab 3"])
        assert len(result.chapters) == 3

    def test_empty_list(self):
        result = ChapterIdentification(chapters=[])
        assert result.chapters == []


# ------------------ MindmapResponse ------------------
class TestMindmapResponse:
    def test_valid(self):
        result = MindmapResponse(
            title="Test",
            mermaid="mindmap\n  root((Test))",
            sources=["source1"],
        )
        assert result.title == "Test"

    def test_empty_sources(self):
        result = MindmapResponse(title="T", mermaid="m", sources=[])
        assert result.sources == []


class TestMindmapNode:
    def test_valid(self):
        node = MindmapNode(label="Parent", children=[
            MindmapNode(label="Child1"),
            MindmapNode(label="Child2"),
        ])
        assert len(node.children) == 2

    def test_default_children(self):
        node = MindmapNode(label="Solo")
        assert node.children == []


# ------------------ MCQResponse ------------------
class TestMCQOption:
    def test_valid_labels(self):
        for label in ["A", "B", "C", "D"]:
            option = MCQOption(label=label, text=f"Option {label}")
            assert option.label == label

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            MCQOption(label="E", text="Invalid")


class TestMCQQuestion:
    def test_valid(self):
        question = MCQQuestion(
            question="What is Python?",
            options=[
                MCQOption(label="A", text="Language"),
                MCQOption(label="B", text="Snake"),
                MCQOption(label="C", text="Tool"),
                MCQOption(label="D", text="OS"),
            ],
            correct_answer="A",
            explanation="Python is a programming language.",
        )
        assert question.correct_answer == "A"


class TestMCQResponse:
    def test_valid(self):
        result = MCQResponse(
            topic="Python",
            difficulty="easy",
            questions=[
                MCQQuestion(
                    question="Q1",
                    options=[
                        MCQOption(label="A", text="a"),
                        MCQOption(label="B", text="b"),
                        MCQOption(label="C", text="c"),
                        MCQOption(label="D", text="d"),
                    ],
                    correct_answer="A",
                    explanation="exp",
                )
            ],
            sources=["src1"],
        )
        assert result.topic == "Python"
        assert len(result.questions) == 1


# ------------------ EssayResponse ------------------
class TestEssayQuestion:
    def test_valid(self):
        q = EssayQuestion(
            question="Explain RAG",
            key_points=["Retrieval", "Generation"],
            explanation="Tests understanding of RAG.",
        )
        assert len(q.key_points) == 2


class TestEssayResponse:
    def test_valid(self):
        result = EssayResponse(
            topic="RAG",
            difficulty="medium",
            questions=[
                EssayQuestion(
                    question="Q1",
                    key_points=["k1"],
                    explanation="exp",
                )
            ],
            sources=["src"],
        )
        assert result.topic == "RAG"


# ------------------ ChatPayload ------------------
class TestChatPayload:
    def test_valid(self):
        payload = ChatPayload(
            user_id="u1",
            session_id="s1",
            book_id="b1",
            messages=[Message(role="user", content="Hello")],
        )
        assert payload.user_id == "u1"
        assert payload.session_id == "s1"
        assert payload.book_id == "b1"
        assert len(payload.messages) == 1


# ------------------ TOC Schemas ------------------
class TestTOCDetection:
    def test_valid(self):
        result = TOCDetection(is_toc=True, thinking="Looks like TOC")
        assert result.is_toc is True

    def test_default_thinking(self):
        result = TOCDetection(is_toc=False)
        assert result.thinking == ""


class TestTOCChapter:
    def test_valid(self):
        chapter = TOCChapter(
            number="1",
            title="Introduction",
            page=10,
            subsections=[TOCChapter(number="1.1", title="Background", page=12)],
        )
        assert len(chapter.subsections) == 1

    def test_defaults(self):
        chapter = TOCChapter(title="Solo")
        assert chapter.number is None
        assert chapter.page is None
        assert chapter.subsections == []


class TestTOCContent:
    def test_valid(self):
        content = TOCContent(
            toc_text="1. Chapter 1\n2. Chapter 2",
            chapters=[TOCChapter(title="Chapter 1")],
        )
        assert len(content.chapters) == 1


class TestPageIndexDetection:
    def test_valid(self):
        result = PageIndexDetection(page_index_given_in_toc=True)
        assert result.page_index_given_in_toc is True
