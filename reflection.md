# Reflection Paper: First Aid Micro-Guide RAG
**Team:** Nitish Kannan (vqa8ue), Keegan Jewell (mmr2ve)
**Date:** December 15, 2025

## A. Architecture & Design Decisions
For this capstone, we built a domain-specific RAG system focused on First Aid. We chose this domain because accurate, offline-capable medical information is critical during emergencies where internet access might be unreliable.

**Model Choices:**
We selected `BAAI/bge-small-en-v1.5` for our embedding model. This model offers an excellent balance of speed and accuracy (384 dimensions), which is essential for running on a CPU-only environment like the requested 16GB VM. We avoided larger models to ensure the API latency remained low (under 1 second).

**Chunking Strategy:**
We used a hybrid chunking strategy. For the CSV data, we treated each row (Symptom/Treatment) as a single distinct document to preserve the structured nature of the data. For the PDF and Text files, we used paragraph-level chunking. This ensures that when a user asks about "Cuts," they get the specific treatment row rather than a fragmented sentence.

## B. Retrieval Quality & Failure Analysis
**Success Case:**
When querying *"How do I treat a cut?"*, the system successfully retrieved the `CUT_MINOR` row from our ETL dataset. The Euclidean distance score was low (indicating high similarity), and the system correctly prioritized this over less relevant FAQ entries.

**Failure Case:**
Early in testing, we found that vague queries like *"It hurts"* performed poorly. The vector search would retrieve random entries containing the word "pain" rather than asking clarifying questions. To mitigate this, we plan to improve the system prompt to encourage the model to state "I need more information" if the distance score is too high.

## C. API & Engineering Challenges
The most significant engineering challenge was the **environment configuration**. We initially attempted to build the system on Google Colab, but we encountered blocking issues where the Flask server would freeze the notebook cell, preventing us from running test queries.

**The Fix:**
We migrated to a containerized environment (GitHub Codespaces) and utilized `faiss-cpu` for vector storage. We also encountered a `ModuleNotFoundError` because the base environment lacked certain build tools, which we resolved by pinning specific versions in `requirements.txt`.

**Latency:**
On the CPU environment, the embedding generation takes approximately 0.4 seconds, and the vector search is instantaneous (<0.01s). The only bottleneck is the initial model loading time at startup (~30 seconds), which we solved by loading the model globally on app launch rather than per request.

## D. Team Collaboration
* **[Your Name]:** Handled the ETL process (cleaning the CSV), the Flask API implementation, and the server setup.
* **[Partner Name (or You if solo)]:** Worked on the RAG pipeline logic (`ingest.py`) and the data gathering.
* We used Git for version control, pushing changes to the `main` branch after verifying they worked in the test environment.