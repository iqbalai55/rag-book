from pydantic import BaseModel, Field
from typing import Optional, List


class TOCDetection(BaseModel):
    is_toc: bool = Field(description="True if the page contains a Table of Contents")
    thinking: str = Field(default="", description="Reasoning about detection")


class TOCChapter(BaseModel):
    number: Optional[str] = Field(default=None, description="Chapter number (e.g., '1', '2.3')")
    title: str = Field(description="Chapter or section title")
    page: Optional[int] = Field(default=None, description="Page number")
    subsections: List["TOCChapter"] = Field(default=[], description="Nested subsections")


class TOCContent(BaseModel):
    toc_text: str = Field(description="Cleaned, formatted TOC text")
    chapters: List[TOCChapter] = Field(default=[], description="Parsed chapter structure")


class PageIndexDetection(BaseModel):
    page_index_given_in_toc: bool = Field(description="True if TOC contains page numbers")
    thinking: str = Field(default="", description="Reasoning about detection")
