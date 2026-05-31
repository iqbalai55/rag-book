from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


class MCQOption(BaseModel):
    label: Literal["A", "B", "C", "D"] = Field(..., description="Option label (A, B, C, or D)")
    text: str = Field(..., description="Option text content")


class MCQQuestion(BaseModel):
    question: str = Field(..., description="The MCQ question text")
    options: List[MCQOption] = Field(..., description="List of 4 options (A, B, C, D)")
    correct_answer: Literal["A", "B", "C", "D"] = Field(..., description="Correct answer label")
    explanation: str = Field(..., description="Brief explanation of the correct answer")


class MCQResponse(BaseModel):
    topic: str = Field(..., description="Topic of the questions")
    difficulty: str = Field(..., description="Difficulty level")
    questions: List[MCQQuestion] = Field(..., description="List of MCQ questions")
    sources: List[str] = Field(..., description="Source references")

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("questions", mode="before")
    @classmethod
    def parse_questions(cls, v):
        if isinstance(v, str):
            import json
            v = json.loads(v)
        return v
    
class EssayQuestion(BaseModel):
    question: str = Field(..., description="Essay question text")
    key_points: List[str] = Field(
        ..., description="Key points expected in a good answer"
    )
    explanation: str = Field(
        ..., description="Explanation why this question is important"
    )

class EssayResponse(BaseModel):
    topic: str = Field(..., description="Topic of the questions")
    difficulty: str = Field(..., description="Difficulty level")
    questions: List[EssayQuestion] = Field(..., description="List of essay questions")
    sources: List[str] = Field(..., description="Source references")

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("questions", mode="before")
    @classmethod
    def parse_questions(cls, v):
        if isinstance(v, str):
            import json
            v = json.loads(v)
        return v