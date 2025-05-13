import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from vector_store import load_vector_store

def calculate_performance_metrics(user_input):
    """
    Calculate performance metrics based on user input and matched data.
    Metrics: Cosine Similarity, Precision, Recall, F1 Score, Relevance Ratio
    """

    # Load the vector store
    vectorizer, vectors, corpus = load_vector_store()

    if not vectorizer:
        return {
            "Cosine Similarity Score": "Vector DB is empty.",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1 Score": "N/A",
            "Relevance Ratio": "N/A"
        }

    # Get the vector for the user input
    q_vector = vectorizer.transform([user_input])
    
    # Cosine similarity calculation
    sim_scores = cosine_similarity(q_vector, vectors)
    top_match = np.argmax(sim_scores)
    matched_text = corpus[top_match]

    # Assuming user feedback for relevance
    # In a real implementation, user feedback would be captured to calculate precision/recall
    feedback = get_user_feedback(user_input, matched_text)  # Function to get user feedback (relevant or not)

    # Cosine Similarity score
    cosine_similarity_score = sim_scores[0][top_match]

    # Calculate Precision, Recall, F1 Score based on user feedback
    # Assume the following:
    # 1 = relevant, 0 = not relevant (we need real feedback from user, which can be added)
    y_true = [1]  # Assuming the correct answer should be relevant
    y_pred = [1 if feedback == 'Yes' else 0]  # Predicted relevance based on feedback

    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)

    # Relevance ratio (percentage of relevant matches found)
    relevance_ratio = (precision * 100)  # Simple approach, but can be expanded based on relevance score

    return {
        "Cosine Similarity Score": cosine_similarity_score,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Relevance Ratio": relevance_ratio
    }

def get_user_feedback(user_input, matched_text):
    """
    Simulate a feedback system for relevance (Yes/No).
    In actual implementation, this should capture real feedback from users.
    """
    # Here we assume that the matched_text is relevant. Replace this with actual feedback collection from the user
    # E.g., via Streamlit: `st.selectbox()` or `st.radio()`

    # For simplicity, assuming 'Yes' for all responses in this simulated function
    return 'Yes'  # This would be dynamic in a real-world scenario