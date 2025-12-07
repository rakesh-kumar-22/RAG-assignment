import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def retrieve_transactions(query, embeddings, texts, top_k=3):
    """
    Returns top_k most relevant transaction texts based on cosine similarity
    
    Args:
        query: User question string
        embeddings: List/array of embedding vectors for all transactions
        texts: List of transaction text strings
        top_k: Number of results to return
    
    Returns:
        List of top_k most relevant transaction texts
    """
    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Get query embedding
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings_array)[0]
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return top_k texts
    return [texts[i] for i in top_indices]
