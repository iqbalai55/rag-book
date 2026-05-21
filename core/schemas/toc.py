from pydantic import BaseModel, Field
from typing import Optional

class PageIndexDetection(BaseModel):
    thinking: str = Field(description="Reasoning about page index detection")
    page_index_given_in_toc: str = Field(description="yes or no")

class TOCDetection(BaseModel):
    is_toc: bool = Field(description="True if the page contains a Table of Contents list.")
    toc_content: Optional[str] = Field(description="The cleaned, formatted TOC text if is_toc is True.")