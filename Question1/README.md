# 📝 Article Generator Showdown using 3 Powerful LLMs

- 🔴 [Live Demo](https://imabhnv-article-generator.streamlit.app/)

This project is a **Streamlit-based intelligent article generator** powered by 3 top-tier open-source and commercial LLMs:

- **Gemma (via Groq SDK)**
- **Gemini (via Google Generative AI SDK)**
- **Cohere Command-R**

🔍 After generation, articles are **automatically evaluated by LLaMA3 (70B)** to decide which model performed the best based on structure, coherence, facts, and style.

---

## 🚀 Features

- Generate detailed articles on **any topic entered by the user**.
- Automatically compare outputs from all 3 models using **LLaMA3 evaluator**.
- Clean and powerful **Streamlit interface** — no dropdowns or manual model selection.
- Real-time **AI-powered verdict** on the best model for every topic.

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/imabhnv/article-generator-chatBot.git
cd article-generator-chatBot
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys Required

You need the following API keys to run this project:

- **Groq API Key** (for Gemma + LLaMA3)
- **Google Generative AI API Key** (for Gemini)
- **Cohere API Key** (for Cohere Command-R)

Use Streamlit secrets for key security. Add this to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
GOOGLE_API_KEY = "your-google-api-key"
COHERE_API_KEY = "your-cohere-api-key"
```

---

## 📜 How to Run

### Run the Streamlit App

```bash
streamlit run app.py
```

This will launch a web app where you enter a topic, and it generates + compares article outputs automatically.

---

### Run the Evaluation Script (Optional)

```bash
python model_evaluation.py
```

This script:
- Uses 3 sample prompts
- Generates articles from each LLM
- Asks LLaMA3 to declare a winner for each

---

## 📂 Project Structure

```
├── app.py                   # Main Streamlit app for article generation + evaluation
├── model_evaluation.py      # Optional script for batch evaluation using prompts
├── requirements.txt         # All Python dependencies
└── README.md                # Project documentation
```

---

## 🙌 Acknowledgements

- [Groq](https://groq.com/) for ultra-fast inference
- [Google Generative AI](https://ai.google.dev/)
- [Cohere AI](https://cohere.com/)

---

## ✨ Made with 💻❤️ by Abhinav Varshney
