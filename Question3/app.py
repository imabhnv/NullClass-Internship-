#  IF YOU WANT TO RUN THIS PROGRAM LOCALLY THEN GO WITH THIS CODE BELOW 
# import streamlit as st
# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import os
# import gdown

# from optimize_cpu import (
#     optimize_memory, 
#     load_optimized_embedding_model,
#     generate_embeddings_batched,
#     load_optimized_llm,
#     check_model_compatibility,
# )

# # 📌 Model Choices
# available_models = [
#     "EleutherAI/gpt-neo-125M",
#     "EleutherAI/pythia-70m",
#     "distilgpt2"
# ]

# # ================== 🔁 CACHES ====================

# @st.cache_data(show_spinner=False)
# def load_data():
#     file_path = 'data/output.json'
    
#     if not os.path.exists(file_path):
#         st.warning("Dataset not found locally. Downloading from Google Drive...", icon="📥")
#         # Replace this with your actual file ID
#         file_id = "1bGPMru_zbwCd06JdTXyacjw2LHWVwqCm"  # Your Google Drive file ID
#         url = f"https://drive.google.com/uc?id={file_id}"
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         gdown.download(url, file_path, quiet=False)
    
#     df = pd.read_json(file_path)
#     df = df[['title', 'abstract', 'categories']]
#     return df

# @st.cache_resource(show_spinner=False)
# def get_embeddings(texts):
#     model = load_optimized_embedding_model("paraphrase-MiniLM-L3-v2")
#     embeddings = generate_embeddings_batched(texts, model, batch_size=32)
#     optimize_memory()
#     return embeddings, model

# @st.cache_resource(show_spinner=False)
# def load_llm(model_name):
#     _, is_compatible = check_model_compatibility(model_name)
#     if not is_compatible:
#         st.warning(f"⚠️ {model_name} might be too large for CPU-only. Try a smaller one.", icon="⚠️")
#     model, tokenizer = load_optimized_llm(model_name)
#     return model, tokenizer

# # ================== 🔍 RAG Functions ====================

# def retrieve_documents(query, df, embeddings, embedding_model, top_k=10):
#     query_embedding = embedding_model.encode([query])[0]
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return df.iloc[top_indices], similarities[top_indices]

# def generate_response(query, context, model, tokenizer):
#     prompt = f"""You are a helpful domain expert assistant. 
# Answer the following question based on the scientific papers provided.
# If you don't know the answer, say so rather than making something up.

# QUESTION: {query}

# RELEVANT PAPERS:
# {context}

# ANSWER:
# """
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=128, 
#             temperature=0.7,
#             do_sample=True,
#             top_p=0.9,
#             num_beams=1,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response.split("ANSWER:")[-1].strip() if "ANSWER:" in response else response

# # ================== 🎯 Streamlit UI ====================

# st.set_page_config(page_title="ArXiv Research Assistant", layout="wide")
# st.title("📚 ArXiv Research Assistant Chatbot")
# st.markdown("Ask questions and get answers based on scientific papers in your dataset.")

# # Sidebar: Model Selection
# st.sidebar.header("⚙️ Configuration")
# selected_model = st.sidebar.selectbox("Select LLM model", available_models)

# # Load dataset & embeddings
# with st.spinner("Loading data and generating embeddings..."):
#     df = load_data()
#     embeddings, embedding_model = get_embeddings(df['abstract'].tolist())
#     model, tokenizer = load_llm(selected_model)

# # Tabs for interface
# tab1, tab2 = st.tabs(["🧠 Summarize & Explain Topic","📄 View Relevant Papers"])

# # Tab 3: Summarize & Explain by Topic
# with tab1:
#     st.subheader("🧠 Summarize & Explain Research Topic")
#     user_topic = st.text_input("Enter a topic or keyword (e.g. LLM Optimization):")

#     if st.button("Summarize & Explain"):
#         if user_topic.strip() == "":
#             st.warning("Please enter a topic.", icon="⚠️")
#         else:
#             with st.spinner("Retrieving relevant papers and generating explanation..."):
#                 top_papers, _ = retrieve_documents(user_topic, df, embeddings, embedding_model)
#                 st.session_state['top_papers'] = top_papers  # ✅ Store in session
#                 abstracts = top_papers['abstract'].tolist()
#                 combined_context = "\n".join(abstracts)

#                 prompt = f"""You are a helpful assistant. 
# Summarize and explain the key points of the following scientific paper abstracts in simple terms.

# PAPER ABSTRACTS:
# {combined_context}

# EXPLANATION:"""
#                 inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#                 with torch.no_grad():
#                     output = model.generate(
#                         **inputs, 
#                         max_new_tokens=128, 
#                         temperature=0.7,
#                         do_sample=True,
#                         top_p=0.9,
#                         num_beams=1,
#                         pad_token_id=tokenizer.eos_token_id
#                     )
#                 explanation = tokenizer.decode(output[0], skip_special_tokens=True)

#                 st.markdown("### 🧠 Explanation")
#                 st.success(explanation)

# with tab2:
#     st.subheader("📄 Top 10 Relevant Papers (from your last query)")
#     if 'top_papers' in st.session_state:
#         top_papers = st.session_state['top_papers']
#         for i, row in top_papers.iterrows():
#             st.markdown(f"**{i+1}. Title:** {row['title']}")
#             st.markdown(f"**Abstract:** {row['abstract']}")
#             st.markdown(f"**Categories:** `{row['categories']}`")
#             st.markdown("---")
#     else:
#         st.info("Ask a question in the first tab to see results here.")


# HERE I'M USING THIS CODE BECAUSE OUTPUT.JSON IS NOT RECOGNIZED BY STREAMLIT CLOUD SO I HAVE TO IMPORT IT FROM GOOGLE DRIVE BUT YOU CAN USE ABOVE CODE ON YOUR SYSTEM
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

from optimize_cpu import (
    optimize_memory, 
    load_optimized_embedding_model,
    generate_embeddings_batched,
    load_optimized_llm,
    check_model_compatibility,
)

# 📌 Model Choices
available_models = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-70m",
    "distilgpt2"
]

# ================== 🔁 CACHES ====================

@st.cache_data(show_spinner=False)
def load_data():
    file_path = 'data/output.json'
    
    if not os.path.exists(file_path):
        st.warning("Dataset not found locally. Downloading from Google Drive...", icon="📥")
        # Replace this with your actual file ID
        file_id = "1bGPMru_zbwCd06JdTXyacjw2LHWVwqCm"  # Your Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        gdown.download(url, file_path, quiet=False)
    
    df = pd.read_json(file_path)
    df = df[['title', 'abstract', 'categories']]
    return df

@st.cache_resource(show_spinner=False)
def get_embeddings(texts):
    model = load_optimized_embedding_model("paraphrase-MiniLM-L3-v2")
    embeddings = generate_embeddings_batched(texts, model, batch_size=32)
    optimize_memory()
    return embeddings, model

@st.cache_resource(show_spinner=False)
def load_llm(model_name):
    _, is_compatible = check_model_compatibility(model_name)
    if not is_compatible:
        st.warning(f"⚠️ {model_name} might be too large for CPU-only. Try a smaller one.", icon="⚠️")
    model, tokenizer = load_optimized_llm(model_name)
    return model, tokenizer

# ================== 🔍 RAG Functions ====================

def retrieve_documents(query, df, embeddings, embedding_model, top_k=10):
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices], similarities[top_indices]

def generate_response(query, context, model, tokenizer):
    prompt = f"""You are a helpful domain expert assistant. 
Answer the following question based on the scientific papers provided.
If you don't know the answer, say so rather than making something up.

QUESTION: {query}

RELEVANT PAPERS:
{context}

ANSWER:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128, 
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("ANSWER:")[-1].strip() if "ANSWER:" in response else response

# ================== 🎯 Streamlit UI ====================

st.set_page_config(page_title="ArXiv Research Assistant", layout="wide")
st.title("📚 ArXiv Research Assistant Chatbot")
st.markdown("Ask questions and get answers based on scientific papers in your dataset.")

# Sidebar: Model Selection
st.sidebar.header("⚙️ Configuration")
selected_model = st.sidebar.selectbox("Select LLM model", available_models)

# Load dataset & embeddings
with st.spinner("Loading data and generating embeddings..."):
    df = load_data()
    embeddings, embedding_model = get_embeddings(df['abstract'].tolist())
    model, tokenizer = load_llm(selected_model)

# Tabs for interface
tab1, tab2 = st.tabs(["🧠 Summarize & Explain Topic","📄 View Relevant Papers"])

# Tab 3: Summarize & Explain by Topic
with tab1:
    st.subheader("🧠 Summarize & Explain Research Topic")
    user_topic = st.text_input("Enter a topic or keyword (e.g. LLM Optimization):")

    if st.button("Summarize & Explain"):
        if user_topic.strip() == "":
            st.warning("Please enter a topic.", icon="⚠️")
        else:
            with st.spinner("Retrieving relevant papers and generating explanation..."):
                top_papers, _ = retrieve_documents(user_topic, df, embeddings, embedding_model)
                st.session_state['top_papers'] = top_papers  # ✅ Store in session
                abstracts = top_papers['abstract'].tolist()
                combined_context = "\n".join(abstracts)

                prompt = f"""You are a helpful assistant. 
Summarize and explain the key points of the following scientific paper abstracts in simple terms.

PAPER ABSTRACTS:
{combined_context}

EXPLANATION:"""
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                with torch.no_grad():
                    output = model.generate(
                        **inputs, 
                        max_new_tokens=128, 
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                explanation = tokenizer.decode(output[0], skip_special_tokens=True)

                st.markdown("### 🧠 Explanation")
                st.success(explanation)

with tab2:
    st.subheader("📄 Top 10 Relevant Papers (from your last query)")
    if 'top_papers' in st.session_state:
        top_papers = st.session_state['top_papers']
        for i, row in top_papers.iterrows():
            st.markdown(f"**{i+1}. Title:** {row['title']}")
            st.markdown(f"**Abstract:** {row['abstract']}")
            st.markdown(f"**Categories:** `{row['categories']}`")
            st.markdown("---")
    else:
        st.info("Ask a question in the first tab to see results here.")
