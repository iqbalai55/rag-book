from pydantic import BaseModel, Field
from typing import List


class ExpertiseDetection(BaseModel):
    domain: str = Field(description="Domain utama (e.g., Software Engineering, Sejarah, Biologi)")
    sub_fields: List[str] = Field(description="Sub-bidang spesialisasi (2-5 item)")
    expertise_prompt: str = Field(
        description="Deskripsi keahlian untuk system prompt, 1-2 kalimat dalam Bahasa Indonesia"
    )
    book_type: str = Field(description="Tipe buku: textbook, populer, referensi, teknis, dll")
