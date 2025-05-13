import streamlit as st
import google.generativeai as genai
import cohere
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from groq import Groq 

# Download NLTK Resources
nltk.download('stopwords')
nltk.download('wordnet')

# ===== Set Page Config =====
st.set_page_config(page_title="📝 Article Generator using 3 Powerful LLMs", layout="wide")

# ===== NLP Preprocessing Functions =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ====== API KEYS PLACEHOLDER (Insert Your Keys Here) =======
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyBz830luF6uDWgqht5ngyw34l-KWNxUXr0"
COHERE_API_KEY = "kteKmqxtDuJhLHfkfWazGPfkZUu81u3b67ux124S"

# ====== Initialize Clients =======
groq_client = Groq(api_key=GROQ_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)

# ====== LLM Functions =======
def generate_with_groq(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        if chat_completion.choices:
            return chat_completion.choices[0].message.content
        else:
            return "Error: No response generated."
    except Exception as e:
        return f"Error occurred: {str(e)}"

def generate_with_google(prompt):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error occurred: {str(e)}"

def generate_with_cohere(prompt):
    try:
        response = cohere_client.chat(
            model="command-r",  
            message=prompt,
            temperature=0.7,
            max_tokens=500
        )
        return response.text
    except Exception as e:
        return f"Error occurred: {str(e)}"

# ====== Streamlit UI =======
st.title("📝 Intelligent Article Generator Chatbot")
st.markdown("""---""")

selected_model = st.selectbox(
    "Select a Model to Generate Article:",
    ["GROQ (DeepSeek Llama)", "Google Generative AI (Gemini-Pro)", "Cohere (Command-R)"]
)

user_topic = st.text_input("Enter Topic for Article:")

if st.button("Generate Article"):
    if user_topic:
        with st.spinner('Generating your article, please wait... 🚀'):
            processed_topic = preprocess_text(user_topic)
            final_prompt = f"Write a detailed and informative article on: {processed_topic}"

            if "GROQ" in selected_model:
                article = generate_with_groq(final_prompt)
            elif "Google" in selected_model:
                article = generate_with_google(final_prompt)
            else:
                article = generate_with_cohere(final_prompt)

            st.success("Article Generated Successfully! ✅")
            st.markdown("### ✨ Here is your Article:")
            st.write(article)
    else:
        st.error("⚠️ Please enter a topic!")

st.markdown("""---
Made with ❤️ by Abhinav Varshney 🚀""")