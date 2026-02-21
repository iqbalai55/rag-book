from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    session_id: str
    collection_name: str
    messages: List[Message]