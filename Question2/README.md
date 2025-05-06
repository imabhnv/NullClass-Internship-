# 🧠 JARVIS: Dynamic Vector-Based Chatbot

- [Live App](https://imabhnv-dynamic-vectorchatbot.streamlit.app/)
- A smart chatbot that answers based purely on the data provided by the user. It fetches domain-specific news, stores it as memory, and responds to queries strictly using that memory — no assumptions, no hallucinations.

## 🚀 Features

- Chat directly with the memory-based chatbot.
- Add new information manually via the interface.
- Fetch live news articles on any topic and store them in memory.
- Memory is updated dynamically and converted into vector embeddings.
- Chatbot only responds using the stored memory.

## 🧩 How Does This Work?

1. Run the `app.py` file to launch the Streamlit interface.
2. You can:
   - Ask questions based on the chatbot’s memory (`data.txt`).
   - Add custom knowledge manually.
   - Fetch news articles of your interest using the API and auto-store them.
3. Data is converted into vectors using the vector store.
4. Chatbot retrieves the closest matching content using cosine similarity and formulates a response based strictly on that.

## 📁 Project Structure

```
├── app.py                    # Main Streamlit app
├── vector_store.py           # Handles vector generation and retrieval
├── data.txt                  # Stores chatbot memory
├── performance_metrics.py    # Evaluates similarity-based retrieval accuracy
├── requirements.txt          # Project dependencies
```

## ✅ Performance Evaluation

- Calculates cosine similarity between queries and memory.
- Matches model-predicted response with the actual context retrieved.
- Outputs match % to evaluate retrieval performance.


## 🧠 Notes

- No external knowledge is used — answers are memory-based.
- Ensure your API keys are added in the `app.py` file.
- If the memory is empty or irrelevant, the model is instructed to respond accordingly.

---

Made with 💻♥️ by Abhinav Varshney🚀
