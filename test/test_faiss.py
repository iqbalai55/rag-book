import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.book_faiss_agent import BookAgent

agent = BookAgent(faiss_path="faiss_index_hp")

response = agent.ask("How agile in software development works?")

print("Answer:")
print(response["answer"])

print("Pages:")
print(response["pages"])
