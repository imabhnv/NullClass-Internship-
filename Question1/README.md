
# 📝 Article Generator Showdown using 3 Powerful LLMs

- 🔴 [Live Demo](https://imabhnv-article-generator.streamlit.app/)

This project is a **Streamlit-based intelligent article generator and evaluator** powered by 3 top-tier LLMs:

- **Gemma (via Groq SDK)**
- **Gemini-Pro (via Google Generative AI SDK)**
- **Cohere Command-R**

🔍 After generation, articles from all 3 models are **automatically evaluated by LLaMA3 (70B)** based on relevance, structure, depth, clarity, and coherence.

---

## 🚀 Features

- Generate detailed articles on **any topic** using a selected LLM.
- Compare article quality from all 3 LLMs using a **LLaMA3-powered evaluator**.
- Clean, tab-based **Streamlit interface** — one for article generation, one for evaluation.
- Real-time, **AI-generated scoring and winner analysis**.

---

## 🔑 API Keys Required

Add the following keys to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
GOOGLE_API_KEY = "your-google-api-key"
COHERE_API_KEY = "your-cohere-api-key"
```

These are used for:
- **Gemma & LLaMA3** → Groq
- **Gemini-Pro** → Google Generative AI
- **Command-R** → Cohere

---

You'll get a 2-tab interface:
- **Generate Article** using any one model.
- **Compare All Models** to find the best using LLaMA3.

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit app (UI + model evaluation logic)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🙌 Acknowledgements

- [Groq](https://groq.com/) 
- [Google Generative AI](https://ai.google.dev/)
- [Cohere](https://cohere.com/)
- [Streamlit](https://streamlit.io/)

---

## ✨ Made with 💻❤️ by Abhinav Varshney
