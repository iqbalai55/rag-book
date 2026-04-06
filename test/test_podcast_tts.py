import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from schemas.podcast import PodcastScriptResponse
from services.audio.tts.tts_engine import generate_tts_podcast

json_path = r"test\test_tts.json"

# 📥 load json
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ parse ke schema
try:
    script = PodcastScriptResponse(**data)
except Exception as e:
    print("❌ Invalid JSON format:", e)

# 🎙️ generate audio
file_path = generate_tts_podcast(script)

print("🎧 Audio generated:", file_path)