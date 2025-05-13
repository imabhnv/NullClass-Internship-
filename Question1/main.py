import streamlit as st
import google.generativeai as genai
import cohere
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from groq import Groq

# Download NLTK Resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set Page Config
st.set_page_config(page_title="📝 Article Generator using 3 Powerful LLMs", layout="wide")

# ===== NLP Preprocessing Function =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# API Keys
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# ===== Model Functions =====
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

# ===== Streamlit UI =====
st.title("📝 Intelligent Article Generator Chatbot")
st.markdown("---")

tab1, tab2 = st.tabs(["🧠 Generate Article", "📊 Compare LLMs"])

with tab1:
    selected_model = st.selectbox("Select a Model to Generate Article:", ["Gemma", "Gemini-Pro", "Command-R"])
    user_topic = st.text_input("Enter Topic for Article:")

    if st.button("Generate Article"):
        if user_topic:
            with st.spinner('Generating your article, please wait... 🚀'):
                processed_topic = preprocess_text(user_topic)
                final_prompt = f"Write a detailed and informative article with proper conclusion on: {processed_topic}"

                if selected_model == "Gemma":
                    article = generate_with_groq(final_prompt)
                elif selected_model == "Gemini-Pro":
                    article = generate_with_google(final_prompt)
                else:
                    article = generate_with_cohere(final_prompt)

                st.success("Article Generated Successfully! ✅")
                st.markdown("### ✨ Here is your Article:")
                st.write(article)
        else:
            st.error("⚠️ Please enter a topic!")

with tab2:
    eval_topic = st.text_input("Enter a Topic to Compare All Models:")
    if st.button("Run LLM Comparison"):
        if eval_topic:
            with st.spinner("Running comparison..."):
                processed = preprocess_text(eval_topic)
                prompt_text = f"Write a detailed article about: {processed}"

                google_article = generate_with_google(prompt_text)
                groq_article = generate_with_groq(prompt_text)
                cohere_article = generate_with_cohere(prompt_text)

                evaluation = judge_articles_llama3(google_article, groq_article, cohere_article, eval_topic)

                st.markdown("### 🔍 Evaluation Report:")
                st.markdown(evaluation)
        else:
            st.error("⚠️ Please enter a topic for evaluation!")

st.markdown("""---  
Made with ❤️ by Abhinav Varshney 🚀""")
