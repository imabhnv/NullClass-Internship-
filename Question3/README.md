
# 🤖 ArXiv Research Assistant Chatbot 📚

- [Live App](https://imabhnv-arxiv-chatbot.streamlit.app/)
- This project allows users to query a chatbot based on scientific papers related to computer science from the ArXiv dataset. The chatbot leverages NLP techniques, open-source LLMs, and Streamlit to provide answers and explanations.

## 🔁 Project Workflow

### 1️⃣ Preprocessing the ArXiv Dataset 🧹
The dataset used for this project is the `arxiv-metadata-oai-snapshot.json` file, which contains metadata of scientific papers. Before running the chatbot, we need to preprocess the dataset to filter out the relevant papers based on the field of computer science.

Run the following script to filter and preprocess the dataset:(Assuming you have downloaded the metadata file from Kaggle and saved in data folder)
```bash
python process_arxiv.py --input data/arxiv-metadata-oai-snapshot.json  --output data/output.json --domain cs --max_papers 1000
```
This will generate the `output.json` file 📄 containing the filtered papers relevant to computer science, which will be used for generating embeddings and querying by the chatbot.

### 2️⃣ Optimizing CPU Memory 🧠💻
To efficiently use models on CPU, we have the `optimize_cpu.py` file that optimizes memory usage and helps in selecting models that are compatible with CPU. The following functions are included in this file:

- 🔄 **optimize_memory**: Clears garbage collection and frees GPU memory cache (if any).
- 📊 **get_system_memory_info**: Fetches system memory info.
- 📦 **get_embedding_batch_size**: Adjusts batch size for embeddings dynamically.
- 🧠 **load_optimized_embedding_model**: Loads SentenceTransformer for CPU.
- 🧬 **generate_embeddings_batched**: Efficient batch embedding generator.
- 🤖 **load_optimized_llm**: Loads lightweight LLMs like `facebook/opt-125m`.
- ✅ **check_model_compatibility**: Checks CPU compatibility.

### 3️⃣ Running the Chatbot Application 🚀💬
Once the dataset is preprocessed, the `app.py` file is used to run the chatbot application. This is a Streamlit-based web interface that interacts with the user, generates embeddings, retrieves relevant documents, and generates answers using a lightweight LLM.

Run the following command to start the application:
```bash
streamlit run app.py
```

The application will allow you to:
- 🎛️ Select an LLM model from available options.
- 🔍 Input a research topic or keyword.
- 📑 Retrieve relevant papers from the dataset.
- 🧠 Summarize and explain the research topic.

## 📂 File Overview
- `process_arxiv.py` 📄: Preprocesses ArXiv dataset.
- `optimize_cpu.py` ⚙️: Optimizes model usage on CPU.
- `app.py` 💬: Main chatbot application with Streamlit.

## 📦 Requirements
- Python 3.7+ 🐍
- Required Libraries: 
  - pandas 📊
  - numpy ➕
  - torch 🔥
  - transformers 🤖
  - sentence-transformers 🧠
  - sklearn 📚
  - streamlit 🌐
  - psutil 📈

Install with:
```bash
pip install -r requirements.txt
```

## 📝 Notes
- ✅ Make sure you have access to the ArXiv dataset.
- 📈 Chatbot quality depends on embeddings and LLM.
- 💾 8GB+ RAM recommended for smooth experience.
- 💻 Three different models can generate different results from each other.
- 😁 I also tried to add relevant emojis and comments in files.

## Made with 💻♥️ by Abhinav Varshney🚀
---
