import os
import uuid
import numpy as np
import torch
import soundfile as sf
from dotenv import load_dotenv

from typing import List, Literal
from pydantic import BaseModel

from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from schemas.podcast import PodcastScriptResponse

# ⚙️ CONFIG
MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
CHECKPOINT_FILENAME = "t3_cfg.safetensors"
OUTPUT_DIR = "audio"
SAMPLE_RATE = 24000

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 🚀 LOAD MODEL (ONCE)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading TTS model...")
model = ChatterboxTTS.from_pretrained(device=device)

checkpoint_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=CHECKPOINT_FILENAME,
    token=HF_TOKEN
)

t3_state = load_file(checkpoint_path, device="cpu")
model.t3.load_state_dict(t3_state)

torch.cuda.empty_cache()
print("✅ TTS model ready!")


# 🔊 LOW-LEVEL TTS (SINGLE TEXT)
def generate_tts(
    text: str,
    audio_prompt_path: str = None
):
    """
    Generate audio untuk satu kalimat
    Return: numpy array (wav)
    """
    with torch.no_grad():
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path
        )
    return wav


def generate_tts_podcast(
    script: PodcastScriptResponse,
    audio_prompt_host: str = r'services\audio\tts\host_audio.wav',
    audio_prompt_guest: str = r'services\audio\tts\guest_audio.wav'
) -> str:
    """
    Generate full podcast audio tanpa silence
    """

    audio_chunks = []

    for turn in script.dialogue:
        speaker = turn.speaker
        text = turn.text

        # 🎙️ pilih voice
        audio_prompt = (
            audio_prompt_host if speaker == "Host"
            else audio_prompt_guest
        )

        # 🔥 generate audio
        wav = generate_tts(text, audio_prompt)

        # 🔥 ensure numpy + 1D
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        if wav.ndim > 1:
            wav = wav.squeeze()

        audio_chunks.append(wav)

    # 🔥 gabungkan semua audio (tanpa silence)
    final_audio = np.concatenate(audio_chunks)

    # 🔥 normalize
    if np.max(np.abs(final_audio)) > 0:
        final_audio = final_audio / np.max(np.abs(final_audio))

    # 💾 save
    file_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.wav")
    sf.write(file_path, final_audio, SAMPLE_RATE)

    return file_path
