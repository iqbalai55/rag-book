from langchain.tools import tool
from typing import Tuple, List
from langchain_core.documents import Document
from langchain.agents import create_agent

from prompts.podcast import PODCAST_SCRIPT_PROMPT, PODCAST_SYSTEM_PROMPT
from schemas.podcast import PodcastScriptResponse  

from agents.book_qdrant_agent import BookQdrantAgent

from services.audio.tts.tts_engine import generate_tts_podcast

class BookPodcastAgent(BookQdrantAgent):
    """Extended agent with end-to-end podcast generation."""

    def __init__(self, qdrant_db, course_id: str, checkpointer=None, k: int = 3):
        super().__init__(qdrant_db, course_id, checkpointer, k)

        @tool("generate_complete_podcast", description="Cari materi dari buku, buat naskah, dan hasilkan audio podcast sekaligus.")
        def generate_complete_podcast(
            topic: str,
            duration_minutes: int = 5,
            style: str = "educational"
        ) -> dict:
            
            # RETRIEVE: Ambil konteks dari Qdrant
            context, _, unique_sources = self._retrieve_context(topic)
            if not context:
                return {"status": "error", "message": "Tidak ditemukan materi relevan di buku."}

            # GENERATE SCRIPT: Panggil LLM untuk bikin naskah terstruktur
            prompt = PODCAST_SCRIPT_PROMPT.format(
                topic=topic,
                duration=duration_minutes,
                style=style,
                context=context[:4000]
            )
            
            structured_llm = self.llm.with_structured_output(PodcastScriptResponse)
            script_data: PodcastScriptResponse = structured_llm.invoke(prompt)

            # Melengkapi metadata script
            script_data.topic = topic
            script_data.duration_minutes = duration_minutes
            script_data.style = style
            script_data.sources = unique_sources[:5]

            # GENERATE AUDIO: Langsung proses ke TTS engine
            try:
                audio_url, num_turns = generate_tts_podcast(script_data)
                
                return {
                    "status": "success",
                    "topic": topic,
                    "audio_url": audio_url,
                    "script": script_data.dict(), 
                    "metadata": {
                        "turns": num_turns,
                        "sources": unique_sources[:5]
                    }
                }
            except Exception as e:
                return {
                    "status": "error", 
                    "message": f"Gagal generate audio: {str(e)}",
                    "script_preview": script_data.dict() 
                }

        self.agent = create_agent(
            model=self.llm,
            system_prompt=PODCAST_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            tools=[
                generate_complete_podcast 
            ],
        )