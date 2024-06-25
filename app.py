import google.generativeai as genai
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# Configure Gemini API key
GOOGLE_API_KEY = <"GEMINI API KEY">  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize conversation history
conversation_history = []
# Fetch lecture notes and model architectures
def fetch_lecture_notes():
    lecture_urls = [
        "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
        "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
        "https://stanford-cs324.github.io/winter2022/lectures/data/",
        "https://stanford-cs324.github.io/winter2022/lectures/modeling/"
    ]
    lecture_texts = []
    for url in lecture_urls:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Fetched content from {url}")
            lecture_texts.append((extract_text_from_html(response.text), url))
        else:
            print(f"Failed to fetch content from {url}, status code: {response.status_code}")
    return lecture_texts


def fetch_model_architectures():
    url = "https://github.com/Hannibal046/Awesome-LLM#milestone-papers"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Fetched model architectures, status code: {response.status_code}")
        return extract_text_from_html(response.text), url
    else:
        print(f"Failed to fetch model architectures, status code: {response.status_code}")
        return "", url

# Extract text from HTML content
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator="\n", strip=True)
    return text

# Generate embeddings using SentenceTransformers
def create_embeddings(texts, model):
    texts_only = [text for text, _ in texts]
    embeddings = model.encode(texts_only)
    return embeddings


# Initialize FAISS index
def initialize_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Assuming all embeddings have the same dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# Handle natural language queries
def handle_query(query, faiss_index, embeddings_texts, model, conversation_history):
    query_embedding = model.encode([query]).astype('float32')

    # Search FAISS index
    _, indices = faiss_index.search(query_embedding, 1)  # Retrieve top 1 result
    relevant_text = embeddings_texts[indices[0][0]]

    # Update conversation history
    conversation_history.add_to_history(query, relevant_text)
    
    return relevant_text
