# 🚀 DocQueryAI
### 🧠 Context-Aware Document Question Answering with RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/VectorDB-FAISS-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/LLM-LLaMA3-green?style=for-the-badge">
</p>

<p align="center">
  <b>💡 Retrieve first. Generate second.</b><br>
  <i>Minimizing hallucinations. Maximizing accuracy.</i>
</p>

---

## 🌟 What is DocQueryAI?

**DocQueryAI** is a powerful **Retrieval-Augmented Generation (RAG)** system that transforms how users interact with documents.

Instead of relying on generic LLM responses, it:
- 🔎 Retrieves **relevant context**
- 🧠 Generates **fact-grounded answers**
- 🎯 Ensures **precision and reliability**

---

## ✨ Key Features

| Feature | Description |
|--------|------------|
| 🔍 Semantic Search | Finds meaning, not just keywords |
| ⚡ FAISS Retrieval | Lightning-fast vector similarity search |
| 🧠 LLM Reasoning | Context-aware answers using LLaMA3 |
| 🧩 Smart Chunking | Overlapping chunks for better continuity |
| 🎯 Adaptive Filtering | Removes noise, keeps relevance |
| 📄 Extensible | Works with text, scalable to PDFs & more |

---

## 🏗️ Architecture Overview

```mermaid
   flowchart LR
    A["User Query"]:::user --> B["Embedding Model"]:::embed
    B --> C["FAISS Vector Search"]:::vector
    C --> D["Top-K Chunks"]:::chunks
    D --> E["Adaptive Filtering"]:::filter
    E --> F["Context Builder"]:::context
    F --> G["LLaMA3 (Groq)"]:::llm
    G --> H["Final Answer"]:::output

    classDef user fill:#1e293b,color:#ffffff,stroke:#38bdf8;
    classDef embed fill:#0f172a,color:#22c55e,stroke:#22c55e;
    classDef vector fill:#1f2937,color:#f59e0b,stroke:#f59e0b;
    classDef chunks fill:#111827,color:#eab308,stroke:#eab308;
    classDef filter fill:#1e293b,color:#a78bfa,stroke:#a78bfa;
    classDef context fill:#0f172a,color:#06b6d4,stroke:#06b6d4;
    classDef llm fill:#111827,color:#22c55e,stroke:#22c55e;
    classDef output fill:#1e293b,color:#10b981,stroke:#10b981;
```
