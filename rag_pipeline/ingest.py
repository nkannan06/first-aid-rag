import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INDEX_OUTPUT_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
TEXT_OUTPUT_PATH = os.path.join(BASE_DIR, "indexed_text.csv")

# --- GLOBAL MODEL LOADER (For the API) ---
def load_resources():
    print("⏳ Loading Embedding Model & Index...")
    try:
        # 1. Load Model
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # 2. Load FAISS Index
        if not os.path.exists(INDEX_OUTPUT_PATH):
            raise FileNotFoundError(f"Index not found at {INDEX_OUTPUT_PATH}. Run ingest.py first!")
        
        index = faiss.read_index(INDEX_OUTPUT_PATH)
        
        # 3. Load Text Data
        if not os.path.exists(TEXT_OUTPUT_PATH):
            raise FileNotFoundError(f"Text data not found at {TEXT_OUTPUT_PATH}")
        
        df = pd.read_csv(TEXT_OUTPUT_PATH)
        documents = df['text'].tolist()
        
        print("✅ Resources Loaded Successfully!")
        return {"model": model, "index": index, "documents": documents}
        
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        return None

# --- RETRIEVAL FUNCTION (For the API) ---
def retrieve_context(query, resources, top_k=3):
    if not resources:
        return []
    
    model = resources['model']
    index = resources['index']
    documents = resources['documents']
    
    # Embed the query
    query_vector = model.encode([query], convert_to_numpy=True)
    
    # Search
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = float(distances[0][i])
        
        if idx < len(documents):
            results.append({
                "text": documents[idx],
                "distance": dist,
                "id": "First Aid Database",
                "snippet": documents[idx][:150] + "..."
            })
            
    return results

# --- INGESTION LOGIC (Only runs when you type 'python ingest.py') ---
def load_raw_documents():
    docs = []
    
    # 1. Load CSV
    csv_path = os.path.join(DATA_DIR, "etl_cleaned_dataset.csv")
    if os.path.exists(csv_path):
        print(f"📄 Processing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Combine columns
        df['combined_text'] = df.astype(str).agg(' | '.join, axis=1)
        docs.extend(df['combined_text'].tolist())

    # 2. Load PDF
    pdf_path = os.path.join(DATA_DIR, "First Aid Quick Guide.pdf")
    if os.path.exists(pdf_path):
        print(f"📄 Processing PDF: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(text)
        except Exception as e:
            print(f"⚠️ Error reading PDF: {e}")

    # 3. Load TXT
    txt_path = os.path.join(DATA_DIR, "First_Aid_FAQ_and_Decision_Tips.txt")
    if os.path.exists(txt_path):
        print(f"📄 Processing Text File: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = content.split('\n\n')
            docs.extend([c for c in chunks if c.strip()])

    return docs

def create_index():
    print("🚀 Starting Ingestion Process...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    documents = load_raw_documents()
    
    print(f"📊 Total Documents to Index: {len(documents)}")
    if not documents:
        print("❌ Error: No documents found!")
        return

    print("🧠 Generating Embeddings...")
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    pd.DataFrame({'text': documents}).to_csv(TEXT_OUTPUT_PATH, index=False)
    
    print(f"✅ Success! Index saved to {INDEX_OUTPUT_PATH}")

if __name__ == "__main__":
    create_index()