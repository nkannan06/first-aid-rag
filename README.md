Markdown

# Group Members: Nitish Kannan (vqa8ue), Keegan Jewell (mmr2ve)

# First Aid Micro-Guide RAG System

## Project Overview
This project is a domain specific **Retrieval-Augmented Generation (RAG)** system designed to provide accurate, grounded first-aid advice. Unlike generic chatbots, this system relies exclusively on a curated dataset of medical protocols, ensuring answers are derived from verified sources rather than hallucinated.

The system features:
* **ETL Pipeline:** Ingests and cleans data from CSVs, PDFs, and text files.
* **Vector Search:** Uses FAISS and `BAAI/bge-small-en-v1.5` for high-accuracy semantic retrieval.
* **Flask API:** A robust REST API serving JSON responses for integration with frontends.

---

## Repository Structure

```text
first-aid-rag/
├── api/
│   ├── app.py               # Main Flask Server
│   └── requirements.txt     # Python Dependencies
├── rag_pipeline/
│   ├── ingest.py            # ETL & Embedding Script
│   ├── faiss_index.bin      # Vector Database (Generated)
│   └── indexed_text.csv     # Source Text Lookup (Generated)
├── data/
│   ├── etl_cleaned_dataset.csv          # Structured First Aid Data
│   ├── First Aid Quick Guide.pdf        # Supplementary PDF Protocols
│   └── First_Aid_FAQ_and_Decision_Tips.txt # Common Scenarios
└── reflection.md            # Project Reflection Paper

 Tech Stack & Models
Language: Python 3.12

API Framework: Flask

Embeddings Model: BAAI/bge-small-en-v1.5 (384 dimensions)

Vector Store: FAISS (Facebook AI Similarity Search) - CPU Version

PDF Processing: pypdf

Environment: GitHub Codespaces / Local 16GB VM

 Setup & Installation
1. Prerequisites
Ensure you have Python 3.10+ installed. If running locally, it is recommended to use a virtual environment.

2. Install Dependencies
Open your terminal in the project root and run:

Bash

pip install -r api/requirements.txt
3. Build the Knowledge Base (ETL)
Before running the server, you must ingest the data to create the vector index.

Bash

python rag_pipeline/ingest.py
Expected Output: ✅ Success! Index saved to .../faiss_index.bin

4. Start the API Server
The server must be run from within the api directory to correctly resolve paths.

Bash

cd api
python app.py
The server will start at: http://127.0.0.1:5000

API Documentation
Endpoint: Ask a Question
URL: /api/ask

Method: POST

Content-Type: application/json

Request Format
JSON

{
  "question": "How do I treat a minor burn?"
}
Response Format
JSON

{
  "answer": "Based on the first aid guidelines: Cool the burn under cold running water for at least 10 minutes...",
  "sources": [
    {
      "id": "First Aid Database",
      "snippet": "BURN_MINOR | ... | Cool the burn under cold running water..."
    }
  ]
}
Example Usage (cURL)
Bash

curl -X POST [http://127.0.0.1:5000/api/ask](http://127.0.0.1:5000/api/ask) \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the signs of heat stroke?"}'