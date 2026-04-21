from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    session_id: str
    course_id: str
    messages: List[Message]