from pydantic import BaseModel, Field, field_validator
from typing import List


class ChapterSummary(BaseModel):
    chapter_title: str = Field(..., description="Chapter title")
    summary: str = Field(..., description="Chapter summary")
    key_points: List[str] = Field(..., description="Key points from the chapter")

    @field_validator("key_points", mode="before")
    @classmethod
    def parse_key_points(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class BookSummaryResponse(BaseModel):
    book_id: str = Field(..., description="Book identifier")
    title: str = Field(..., description="Book summary title")
    overview: str = Field(..., description="Book overview")
    chapters: List[ChapterSummary] = Field(..., description="Chapter summaries")
    key_themes: List[str] = Field(..., description="Key themes from the book")
    total_chapters: int = Field(..., description="Total number of chapters")
    sources: List[str] = Field(..., description="Source references")

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("key_themes", mode="before")
    @classmethod
    def parse_key_themes(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class SummaryEditRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    book_id: str = Field(..., description="Book identifier")
    title: str = Field(..., description="Current summary title")
    overview: str = Field(..., description="Current summary overview")
    key_themes: List[str] = Field(..., description="Current key themes")
    chapters: List[ChapterSummary] = Field(..., description="Current chapter summaries")
    instruction: str = Field(..., description="Edit instructions")


class SummaryEditResponse(BaseModel):
    book_id: str = Field(description="Book identifier")
    title: str = Field(description="Modified title")
    overview: str = Field(description="Modified overview")
    key_themes: List[str] = Field(description="Modified key themes")
    chapters: List[ChapterSummary] = Field(description="Modified chapter summaries")
