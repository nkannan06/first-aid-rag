import os
import sys
from flask import Flask, request, jsonify, render_template

# Ensure we can import from rag_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_pipeline.ingest import load_resources, retrieve_context

app = Flask(__name__)

# Load AI Brain once at startup
print("⏳ Starting Server & Loading Resources...")
resources = load_resources()

@app.route('/')
def home():
    # This serves the new HTML file
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('question', '')
    
    if not query:
        return jsonify({"error": "No question provided"}), 400

    # 1. Retrieve relevant info
    results = retrieve_context(query, resources)
    
    # 2. Check if we found anything good (Distance threshold logic)
    # If the best result is too far away (distance > 0.7), we say "I don't know"
    # Note: resources['index'] isn't directly used here, retrieve_context handles it.
    
    if not results or results[0]['distance'] > 0.7:
        return jsonify({
            "answer": "I cannot find information on that in the First Aid Guide. Please call emergency services if this is urgent.",
            "sources": []
        })

    # 3. Formulate Answer (Deterministic RAG)
    # We combine the top result text into a readable answer.
    best_match = results[0]
    answer = f"Based on the first aid guidelines:\n\n{best_match['text']}\n...(See sources for more)"
    
    return jsonify({
        "answer": answer,
        "sources": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)