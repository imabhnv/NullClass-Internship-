import google.generativeai as genai
import cohere
from groq import Groq
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# API keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

prompts = [
    "The impact of artificial intelligence on education"
]

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Model generators
def generate_with_groq(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_with_google(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_with_cohere(prompt):
    try:
        response = cohere_client.chat(
            model="command-r",
            message=prompt,
            temperature=0.7,
            max_tokens=1024
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Judge using LLaMA3
def judge_articles_llama3(google_article, groq_article, cohere_article, topic):
    prompt = f"""
Three different AI models have generated articles on the topic: "{topic}". Evaluate each article based on:
1. Relevance to topic
2. Depth of information
3. Clarity and structure
4. Language fluency
5. Overall coherence

Here are the articles:

🟢 Google Gemini:
\"\"\"{google_article}\"\"\"

🔵 GROQ Gemma:
\"\"\"{groq_article}\"\"\"

🟡 Cohere Command-R:
\"\"\"{cohere_article}\"\"\"

Now provide a detailed evaluation and give a score (out of 10) for each article. Conclude with which article is best and why.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Main evaluation loop
for prompt in prompts:
    print(f"\n🟨 Prompt: {prompt}")

    processed_prompt = preprocess_text(prompt)
    prompt_text = f"Write a detailed article about: {processed_prompt}"

    google_article = generate_with_google(prompt_text)
    groq_article = generate_with_groq(prompt_text)
    cohere_article = generate_with_cohere(prompt_text)

    evaluation = judge_articles_llama3(google_article, groq_article, cohere_article, prompt)
    
    print("\n🧠 Final Evaluation:")
    print(evaluation)
