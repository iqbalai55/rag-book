from pydantic import BaseModel
from typing import List

class DialogueTurn(BaseModel):
    speaker: str  # "Host" or "Guest"
    text: str

class PodcastScriptResponse(BaseModel):
    topic: str
    duration_minutes: int
    style: str
    dialogue: List[DialogueTurn]
    sources: List[str] = []