from pydantic import BaseModel, Field, field_validator
from typing import List


class MindmapNode(BaseModel):
    label: str = Field(description="Node label, max 5-6 words")
    children: List["MindmapNode"] = Field(
        default=[], description="Child nodes"
    )


class MindmapResponse(BaseModel):
    title: str = Field(description="Mindmap root title")
    mermaid: str = Field(description="Mermaid mindmap syntax")
    sources: List[str] = Field(default=[], description="Source references")

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class MindmapEditData(BaseModel):
    title: str = Field(description="Mindmap root title")
    mermaid: str = Field(description="Mermaid mindmap syntax")


class MindmapEditRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    book_id: str = Field(..., description="Book identifier")
    mermaid: str = Field(..., description="Existing mermaid mindmap to edit")
    instruction: str = Field(..., description="Edit instructions")


class MindmapEditResponse(BaseModel):
    book_id: str = Field(description="Book identifier")
    title: str = Field(description="Mindmap title")
    mermaid: str = Field(description="Modified mermaid mindmap")
