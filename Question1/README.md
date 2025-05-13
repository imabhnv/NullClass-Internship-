
# 📝 Article Generator Chatbot using 3 Powerful LLMs

This project provides a **Streamlit**-based chatbot to **generate articles** using three cutting-edge LLMs:

- **GROQ (DeepSeek Llama 70B)**
- **Google Gemini-Pro (via Generative AI SDK)**
- **Cohere Command-R**

Additionally, an **evaluation script** (`model_evaluation.py`) is provided to **compare the performance** of these models using **BLEU score**.

---

## 🚀 Features

- Generate high-quality articles on any topic.
- Choose between three different LLMs.
- Clean and pre-process input topics using NLP techniques.
- Performance evaluation based on BLEU metric.
- Simple and beautiful Streamlit interface.

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/imabhnv/article-generator-chatBot.git
cd article-generator-chatBot
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys Required

You need to insert your API keys in `main.py` and `model_evaluation.py`:

- **Groq API Key**
- **Google Generative AI API Key**
- **Cohere API Key**

Replace the placeholders:

```python
GROQ_API_KEY = "your-groq-api-key"
GOOGLE_API_KEY = "your-google-api-key"
COHERE_API_KEY = "your-cohere-api-key"
```

---

## 📜 How to Run

### For Chatbot UI (Streamlit)

```bash
streamlit run main.py
```

This will launch a web app where you can select a model and input your topic.

---

### For Model Evaluation (Performance Metrics)

```bash
python model_evaluation.py
```

This script will:
- Generate articles for sample prompts.
- Compute BLEU scores.
- Compare models.
- Display the best performing model.

---

## 📊 Evaluation Metric

We use **BLEU Score** to measure the quality of generated text compared to a reference text.  
Higher BLEU scores indicate better performance.

---

## 📂 Project Structure

```
├── main.py                 # Streamlit UI for article generation
├── model_evaluation.py      # Script for model performance comparison
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation
```

---

## 🙌 Acknowledgements

- [Groq SDK](https://groq.com/)
- [Google Generative AI SDK](https://ai.google.dev/)
- [Cohere API](https://cohere.com/)
- [NLTK Library](https://www.nltk.org/)
- [Evaluate Library (Huggingface)](https://huggingface.co/docs/evaluate/index)

---

## ✨ Made with ❤️ by Abhinav Varshney