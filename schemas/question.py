from pydantic import BaseModel, Field
from typing import List, Literal


class MCQOption(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: str


class MCQQuestion(BaseModel):
    question: str
    options: List[MCQOption]
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str


class MCQResponse(BaseModel):
    topic: str
    difficulty: str
    questions: List[MCQQuestion]
    sources: List[str]
    
class EssayQuestion(BaseModel):
    question: str = Field(..., description="Essay question text")
    key_points: List[str] = Field(
        ..., description="Key points expected in a good answer"
    )
    explanation: str = Field(
        ..., description="Explanation why this question is important"
    )


class EssayResponse(BaseModel):
    topic: str
    difficulty: str
    questions: List[EssayQuestion]
    sources: List[str]