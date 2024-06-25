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
def handle_query(query, faiss_index, embeddings_texts, model):
    global conversation_history

    query_embedding = model.encode([query]).astype('float32')

    # Search FAISS index
    _, indices = faiss_index.search(query_embedding, 3)  # Retrieve top 3 results
    relevant_texts = [embeddings_texts[idx] for idx in indices[0]]

    # Combine relevant texts and truncate if necessary
    combined_text = "\n".join([text for text, _ in relevant_texts])
    max_length = 500  # Adjust as necessary
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length] + "..."

    # Generate a response using Gemini
    try:
        response = genai.generate_text(
            model="models/text-bison-001",
            prompt=f"Based on the following context:\n\n{combined_text}\n\nAnswer the following question: {query}",
            max_output_tokens=200
        )
        generated_text = response.result if response else "No response generated."
    except Exception as e:
        print(f"Error generating text: {e}")
        generated_text = "An error occurred while generating the response."

    # Update conversation history
    conversation_history.append(f"User: {query}")
    conversation_history.append(f"System: {generated_text}")

    # Extract sources
    sources = [url for _, url in relevant_texts]

    return generated_text, sources

def generate_concise_response(prompt, context):
    try:
        response = genai.generate_text(
            model="models/text-bison-001",
            prompt=f"{prompt}\n\nContext: {context}\n\nAnswer:",
            max_output_tokens=200
        )
        return response.result if response else "No response generated."
    except Exception as e:
        print(f"Error generating concise response: {e}")
        return "An error occurred while generating the concise response."

# Main function to execute the pipeline
def chatbot(message, history):
    lecture_notes = fetch_lecture_notes()
    model_architectures = fetch_model_architectures()

    all_texts = lecture_notes + [model_architectures]

    # Load the SentenceTransformers model
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embeddings = create_embeddings(all_texts, embedding_model)

    # Initialize FAISS index
    faiss_index = initialize_faiss_index(np.array(embeddings))

    response, sources = handle_query(message, faiss_index, all_texts, embedding_model)
    print("Query:", message)
    print("Response:", response)

    # Format the response with conversation history
    formatted_response = "Conversation History:\n\n"
    for entry in conversation_history:
        formatted_response += entry + "\n"

    formatted_response += "\nCurrent Response:\n" + response

    if sources:
        print("Sources:", sources)
        formatted_response += "\n\nSources:\n" + "\n".join(sources)
    else:
        print("Sources: None of the provided sources were used.")

    # Generate a concise and relevant summary using Gemini
    prompt = "Summarize the user queries so far"
    user_queries_summary = " ".join([entry for entry in conversation_history if entry.startswith("User: ")])
    concise_response = generate_concise_response(prompt, user_queries_summary)
    print("Concise Response:")
    print(concise_response)

    formatted_response += "\n\nConcise Summary:\n" + concise_response

    print("----")

    return formatted_response


iface = gr.ChatInterface(
    chatbot,
    title="LLM Research Assistant",
    description="Ask questions about LLM architectures, datasets, and training techniques.",
    examples=[
        "What are some milestone model architectures in LLMs?",
        "Explain the transformer architecture.",
        "Tell me about datasets used to train LLMs.",
        "How are LLM training datasets cleaned and preprocessed?",
        "Summarize the user queries so far"
    ],
    retry_btn="Regenerate",
    undo_btn="Undo",
    clear_btn="Clear",
)
