
# RAG Book Agent Exploration

This project explores how to build a **good RAG (Retrieval-Augmented Generation) book agent**, allowing us to ask questions using our **book knowledge database**. The goal is to create an agent that can answer queries **accurately and concisely**, while always citing the source and page from the book metadata.

Currently, we have implemented **two approaches**:

- **Qdrant-based RAG agent**
- **FAISS-based RAG agent**

Both approaches allow retrieval from a book vector database and generation of answers using a language model.



## Installation

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd ad-applicability-extractor
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure API Keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=your_selected_model
```


## Usage

Once installed and configured, you can start querying your book database through the RAG agent. Both Qdrant and FAISS approaches provide similar interfaces for:

* Searching book content
* Returning context with **source and page citations**
* Handling conversation history with summarization
* Limiting excessive tool calls for safe usage


