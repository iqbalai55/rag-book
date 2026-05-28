from pydantic import BaseModel, Field
from typing import List


class ChapterIdentification(BaseModel):
    chapters: List[str] = Field(description="List of chapter/topic titles identified from the content")
