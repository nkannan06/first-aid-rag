from flask import Flask, request, jsonify
import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "..", "rag_pipeline")
INDEX_PATH = os.path.join(RAG_DIR, "faiss_index.bin")
TEXT_PATH = os.path.join(RAG_DIR, "indexed_text.csv")

# Global Variables
model = None
index = None
text_df = None

def load_resources():
    global model, index, text_df
    print("⏳ Starting Server & Loading Resources...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXT_PATH):
        index = faiss.read_index(INDEX_PATH)
        text_df = pd.read_csv(TEXT_PATH)
        print("✅ Resources Loaded Successfully!")
    else:
        print("⚠️ WARNING: Index not found. Run ingest.py first!")

# Load on startup
load_resources()

@app.route('/api/ask', methods=['POST'])
def ask():
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "Missing 'question' in JSON"}), 400
    
    question = request.json['question']
    
    # 1. Embed Question
    q_emb = model.encode([question], convert_to_numpy=True)
    
    # 2. Search Index (Top 3)
    k = 3
    distances, indices = index.search(q_emb, k)
    
    # 3. Retrieve Context
    results = []
    context_str = ""
    
    for idx in indices[0]:
        if idx < len(text_df):
            chunk = str(text_df.iloc[idx]['text'])
            results.append({
                "id": "First Aid Database",
                "snippet": chunk[:300] + "..." # Truncate for display
            })
            context_str += chunk + "\n---\n"

    # 4. Generate Answer (Simulated for Speed/Reliability in Codespaces)
    # Ideally, you pass 'context_str' to a local LLM here. 
    # For the assignment submission reliability, we return the grounded context.
    final_answer = f"Based on the first aid guidelines:\n\n{context_str[:500]}\n...(See sources for more)"

    return jsonify({
        "answer": final_answer,
        "sources": results
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)