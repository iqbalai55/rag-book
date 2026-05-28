from pydantic import BaseModel, Field
from typing import List


class MindmapNode(BaseModel):
    label: str = Field(description="Node label, max 5-6 words")
    children: List["MindmapNode"] = Field(
        default=[], description="Child nodes"
    )


class MindmapResponse(BaseModel):
    title: str = Field(description="Mindmap root title")
    mermaid: str = Field(description="Mermaid mindmap syntax")
    sources: List[str] = Field(description="Source references")
