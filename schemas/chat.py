from pydantic import BaseModel
from typing import List

# Simple OpenAI-compatible payload
class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[Message]
    session_id: str = "default"
