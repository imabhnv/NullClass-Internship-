import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from vector_store import load_vector_store, update_vector_store, load_data
from performance_metrics import calculate_performance_metrics  # 🔁 Import added

# CONFIG
st.set_page_config(page_title="🤖 Smart Vector Chatbot")
# st.title("🧠 JARVIS: Dynamic Vector-Based Chatbot")
st.markdown("<h1 style='color: white;font-family: consolas;'>🧠 JARVIS: Dynamic Vector-Based chatBot</h1>", unsafe_allow_html=True)


# KEYS
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# Function to call Gemini with strict instructions
def generate_with_google(prompt):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Function to fetch news and update memory
def fetch_and_generate_news_and_store():
    st.header("🌐 News Data Updater")
    query = st.text_input("Enter the topic/domain to fetch news and store in memory:")

    if st.button("📥 Fetch & Store in Memory"):
        if query.strip() == "":
            st.warning("Please enter a valid topic.")
            return

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])

            if articles:
                formatted_data = ""
                for article in articles[:3]:
                    title = article.get("title", "No Title")
                    description = article.get("description", "No Description")
                    url = article.get("url", "No URL")
                    formatted_data += f"Title: {title}\nDescription: {description}\nURL: {url}\n\n"

                # Save to data.txt
                with open("data.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + formatted_data.strip())

                # Update vector DB
                update_vector_store()
                st.success("✅ News fetched and stored in memory!")
                st.code(formatted_data)
            else:
                st.error("No articles found for this topic.")
        else:
            st.error("❌ Failed to fetch news.")

# Chat section
st.header("💬 Ask Anything Based on Memory")
user_input = st.text_input("Type your question here:")

if user_input:
    st.markdown("#### 🤔 Processing your query...")
    vectorizer, vectors, corpus = load_vector_store()

    if not vectorizer:
        st.error("⚠️ Vector DB is empty. Please update memory first.")
    else:
        q_vector = vectorizer.transform([user_input])
        sim_scores = cosine_similarity(q_vector, vectors)

        # Use full vector memory
        full_context = "\n".join(corpus)

        prompt = f"""
You are a chatbot. Use only the memory data provided below to answer the user's question.
Do not generate or assume anything.else if can't find in the memory then say "I don't have information"

# Memory Data:
# {full_context}

# Question: {user_input}
# Answer:"""

        response = generate_with_google(prompt)
        st.success("🤖 chatBot says: " + response)

        # 🔍 Performance metrics display
        st.subheader("📊 Performance Metrics")
        metrics = calculate_performance_metrics(user_input)
        for key, value in metrics.items():
            st.write(f"**{key}:** {value}")

# Manual memory update
st.header("📝 Manual Memory Update")
with st.expander("📂 View current memory (data.txt)"):
    st.code(load_data(), language="text")

new_data = st.text_area("Add new information manually:", height=150)

if st.button("🔁 Update Memory"):
    if new_data.strip() == "":
        st.warning("Please enter some new information.")
    else:
        with open("data.txt", "a", encoding="utf-8") as f:
            f.write("\n" + new_data.strip())
        update_vector_store()
        st.success("✅ Memory updated and vector DB refreshed!")

# Fetch News and Store
fetch_and_generate_news_and_store()

# Footer
st.markdown("---")
st.caption("Made with ♥️💻 by Abhinav Varshney🚀")