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

# --- MODEL SETUP ---
print("⏳ Loading Embedding Model (BAAI/bge-small-en-v1.5)...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def load_documents():
    docs = []
    
    # 1. Load CSV (Treat all columns as text)
    csv_path = os.path.join(DATA_DIR, "etl_cleaned_dataset.csv")
    if os.path.exists(csv_path):
        print(f"📄 Processing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Combine all columns into one text string per row
        df['combined_text'] = df.astype(str).agg(' | '.join, axis=1)
        docs.extend(df['combined_text'].tolist())

    # 2. Load PDF
    pdf_path = os.path.join(DATA_DIR, "First Aid Quick Guide.pdf")
    if os.path.exists(pdf_path):
        print(f"📄 Processing PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                docs.append(text)

    # 3. Load TXT
    txt_path = os.path.join(DATA_DIR, "First_Aid_FAQ_and_Decision_Tips.txt")
    if os.path.exists(txt_path):
        print(f"📄 Processing Text File: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by double newlines to make chunks
            chunks = content.split('\n\n')
            docs.extend([c for c in chunks if c.strip()])

    return docs

def create_index():
    documents = load_documents()
    print(f"📊 Total Documents/Chunks to Index: {len(documents)}")
    
    if not documents:
        print("❌ Error: No documents found in data/ folder!")
        return

    # Embed Content
    print("🧠 Generating Embeddings...")
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    # Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save Index
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    
    # Save Text (so we can retrieve the actual answer later)
    pd.DataFrame({'text': documents}).to_csv(TEXT_OUTPUT_PATH, index=False)
    
    print(f"✅ Success! Index saved to {INDEX_OUTPUT_PATH}")

if __name__ == "__main__":
    create_index()