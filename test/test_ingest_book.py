import requests
import os

# ---------------- CONFIG ----------------
FASTAPI_URL = "http://localhost:8000/book-qa/ingest"  # change if deployed
PDF_PATH = r"E:\Coding\KitaPandu\rag-book\book\poa.pdf"

# Make sure file exists
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

# ---------------- UPLOAD ----------------
with open(PDF_PATH, "rb") as f:
    files = {
        "file": (os.path.basename(PDF_PATH), f, "application/pdf")
    }
    response = requests.post(FASTAPI_URL, files=files)

# ---------------- RESPONSE ----------------
print("Status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Response content:", response.text)