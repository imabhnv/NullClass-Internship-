import google.generativeai as genai
import cohere
from groq import Groq
import nltk
import evaluate
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

bleu = evaluate.load("bleu")

GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyBz830luF6uDWgqht5ngyw34l-KWNxUXr0"
COHERE_API_KEY = "kteKmqxtDuJhLHfkfWazGPfkZUu81u3b67ux124S"

groq_client = Groq(api_key=GROQ_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

prompts = [
    "The impact of artificial intelligence on education",
    "How climate change is affecting agriculture",
    "The future of electric vehicles in India"
]

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

def generate_with_groq(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
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
            max_tokens=500
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# 📋 Models dictionary
MODELS = {
    "GROQ (DeepSeek)": generate_with_groq,
    "Google Gemini": generate_with_google,
    "Cohere Command-R": generate_with_cohere
}

results = {}

for model_name, generator_function in MODELS.items():
    print(f"\n🔵 Evaluating Model: {model_name}")

    model_results = []
    
    for prompt in prompts:
        processed_prompt = preprocess_text(prompt)
        generated = generator_function(f"Write a detailed article about: {processed_prompt}")
        reference = prompt  
        candidate = generated
        
        bleu_score = bleu.compute(predictions=[candidate], references=[[reference]])["bleu"]
        model_results.append({
            "prompt": prompt,
            "generated_text": generated,
            "bleu_score": round(bleu_score, 4)
        })
    
    results[model_name] = model_results

best_model = None
highest_bleu_score = 0

print("\n🏆 Performance Comparison:")

for model_name, outputs in results.items():
    total_bleu = sum([item["bleu_score"] for item in outputs])
    avg_bleu = total_bleu / len(outputs)
    
    print(f"\n{model_name} - Average BLEU Score: {round(avg_bleu, 4)}")
    
    if avg_bleu > highest_bleu_score:
        highest_bleu_score = avg_bleu
        best_model = model_name

print(f"\n✅ The best model for article generation is: {best_model}")