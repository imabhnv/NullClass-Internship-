import os
import gc
import torch
import psutil
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def optimize_memory():
    """Run garbage collection and clear CUDA cache (if applicable)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_system_memory_info():
    """Get current system memory usage in GB."""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total / (1024**3),
        "available": memory.available / (1024**3),
        "used": memory.used / (1024**3),
        "percent": memory.percent
    }

def get_embedding_batch_size(model_name):
    """Adjust batch size based on available memory and model size."""
    available_gb = get_system_memory_info()["available"]

    if "L3" in model_name:
        base_batch_size = 64
    elif "L6" in model_name:
        base_batch_size = 32
    else:
        base_batch_size = 16

    if available_gb < 2:
        return max(1, base_batch_size // 4)
    elif available_gb < 4:
        return max(1, base_batch_size // 2)
    elif available_gb > 8:
        return base_batch_size * 2
    else:
        return base_batch_size

def load_optimized_embedding_model(model_name="paraphrase-MiniLM-L3-v2"):
    """Load sentence transformer model optimized for CPU."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return SentenceTransformer(model_name)

def generate_embeddings_batched(texts, model, batch_size=None):
    """Generate sentence embeddings in CPU-friendly batches."""
    if batch_size is None:
        batch_size = get_embedding_batch_size(model.__class__.__name__)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        optimize_memory()
    
    return np.vstack(embeddings)

def load_optimized_llm(model_name="facebook/opt-125m"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def check_model_compatibility(model_name):
    """Check if the model is lightweight and suitable for CPU-only usage."""
    cpu_llms = [
        "facebook/opt-125m",
        "EleutherAI/pythia-70m",
        "distilgpt2",
        "gpt2-medium",
        "EleutherAI/gpt-neo-125m"
    ]
    embedding_models = [
        "paraphrase-MiniLM-L3-v2",
        "all-MiniLM-L6-v2",
        "distiluse-base-multilingual-cased-v1"
    ]

    model_name = model_name.lower()
    if any(cpu_model.lower() in model_name for cpu_model in cpu_llms):
        return "llm", True
    elif any(embed_model in model_name for embed_model in embedding_models):
        return "embedding", True
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoModelForCausalLM.config_class.from_pretrained(model_name)
            if hasattr(config, "n_params") and config.n_params < 500_000_000:
                return "llm", True
            else:
                return "llm", False
        except:
            return "unknown", False
