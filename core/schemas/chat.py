from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    user_id: str
    session_id: str
    book_id: str
    messages: List[Message]