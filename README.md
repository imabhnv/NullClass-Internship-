
# 🚀 NullClass Internship Project: Intelligent Chatbots with Open-Source LLMs

This repository contains the complete work done for the NullClass internship project, focused on building intelligent and domain-specific chatbots using cutting-edge NLP techniques, open-source LLMs, and vector databases.

Each part of the project is organized in dedicated folders:
- [`Question1`](./Question1): Article Generator using 3 Open-Source LLMs
- [`Question2`](./Question2): Dynamic Knowledge Base Update System
- [`Question3`](./Question3): ArXiv Research Assistant Chatbot

---

## 📁 Project Structure

### 🔹 Question1 - Article Generator Chatbot

> **Objective**: Compare three open-source LLMs (e.g., LLaMA 2, Mistral, GPT-Neo/OpenChat) and evaluate which one generates the best articles.

**Features**:
- Integrated three LLMs for article generation.
- Text preprocessing using NLP techniques.
- Evaluation based on coherence, fluency, and factual accuracy.
- Performance metrics: BLEU Score, ROUGE Score, Human Evaluation.

**Outcome**:
- Comparative performance report.
- Identification of the best LLM for article creation.

📂 Folder: [`Question1`](./Question1)

---

### 🔹 Question2 - Dynamic Vector Database Updater

> **Objective**: Implement a system that automatically updates the chatbot's knowledge base with fresh content over time.

**Features**:
- Periodic crawling and fetching from data sources.
- Vector DB update using FAISS.
- `update_memory()` mechanism in place.
- Memory-efficient and real-time knowledge ingestion.

**Outcome**:
- Chatbot that learns new content dynamically without retraining.

📂 Folder: [`Question2`](./Question2)

---

### 🔹 Question3 - ArXiv Research Assistant Chatbot

> **Objective**: Build a domain-specific expert chatbot using the [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) (Computer Science subset).

**Features**:
- Preprocessing of large-scale scientific papers dataset.
- Extraction of abstracts, titles, and category metadata.
- RAG-based chatbot using FAISS + Open-source LLM (Mistral/OpenChat).
- Summarization + Explanation of complex research concepts.
- Streamlit GUI with search and Q&A features.

**Outcome**:
- Chatbot that can:
  - Search & summarize research papers.
  - Explain advanced CS concepts.
  - Handle follow-up questions.
  - Serve as an academic assistant.

📂 Folder: [`Question3`](./Question3)

---

## ⚙️ Tech Stack

- Python, Streamlit, FAISS
- Open-Source LLMs: LLaMA 2, Mistral, OpenChat, GPT-Neo
- NLP: NLTK, spaCy, Gensim
- Vector Store: FAISS
- Evaluation Metrics: Precision, Recall, Confusion Matrix, BLEU, ROUGE

---

## ✅ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Evaluation Metrics

Each LLM and chatbot system is evaluated using:
- Accuracy, Precision, Recall (for classification/evaluation)
- Confusion Matrix
- Human-level qualitative analysis
- BLEU/ROUGE scores (for text generation)

---

## 📎 Project Highlights

- ✅ Modular codebase with clean architecture
- ✅ Real-time vector memory updates
- ✅ Streamlit UI for easy interaction
- ✅ Performance comparison of multiple LLMs
- ✅ Handles domain-specific research-level queries

---

## 📌 Author

**Abhinav**  
Intern at NullClass | Aspiring Data Scientist  
GitHub: [@imabhnv](https://github.com/imabhnv)

---

## 🔗 Project Link

🔗 GitHub Repo: [https://github.com/imabhnv/NullClass-Internship-](https://github.com/imabhnv/NullClass-Internship-)

---
