# Lecture-Notes-Chatbot
Intern assessment for ema.ai to develop a chatbot that answers questions about lectures and LLM architectures.

## Model
We are using Gemini due to time constraints and the superior performance it provides over other base open-source models. A better scalable alternative would be fine-tuning an open-source model like sharded LLaMA, but that is planned for future work.

<img width="274" alt="Gemini" src="https://github.com/sambhavnoobcoder/Lecture-Notes-Chatbot/assets/94298612/d5b9dab7-daf2-48e2-ac5b-ddc0b20321a6">

## Task List
This section comprises a checklist of tasks to be completed. They are sorted according to their necessity for the project's completeness.

### Basic Features
- [x] Operating on reference text
- [x] Utilize LLM
- [x] Open Source Vector Indexing
- [x] Open source storage and embedding

### Bonus Features 
- [x] Conversational Memory
- [x] Citing References
- [x] Multiple Class Lecture Notes
- [x] Built a Basic User Interface (Gradio Frontend)
- [x] Deployed Project on Huggingface (Link: <https://huggingface.co/spaces/Sambhavnoobcoder/Lecture-Notes-Chatbot>)

## Project Images
Gradio UI:
<img width="1440" alt="Gradio UI" src="https://github.com/sambhavnoobcoder/Lecture-Notes-Chatbot/assets/94298612/374bb6ad-2af7-4d90-a75d-d27c32722df2">

## Clear and Well-Documented Description of the Approach

### Pipeline Overview

1. **Data Collection**:
   - Lecture notes are fetched from the Stanford CS324 course website.
   - Model architecture information is sourced from the Awesome LLM GitHub repository.

2. **Text Extraction**:
   - HTML content is parsed using BeautifulSoup, removing unnecessary scripts and styles.

3. **Embeddings Generation**:
   - Sentence embeddings are generated using the SentenceTransformers model ('paraphrase-MiniLM-L6-v2').

4. **FAISS Index Initialization**:
   - The embeddings are indexed using FAISS for efficient similarity search.

5. **Query Handling**:
   - User queries are encoded and searched against the FAISS index.
   - Relevant texts are combined and truncated if necessary.

6. **Response Generation**:
   - Responses are generated using the Gemini API, based on the relevant texts and user query.

7. **Conversational Memory**:
   - The system maintains a history of user queries and system responses.

### Decisions on Modules/Frameworks/Models

- **SentenceTransformers**: Chosen for generating high-quality embeddings efficiently.
- **FAISS**: Selected for its fast and scalable similarity search capabilities.
- **BeautifulSoup**: Used for its robust HTML parsing and text extraction.
- **Gradio**: Opted for creating a simple yet effective user interface.
- **Gemini API**: Preferred for its superior performance in generating human-like responses.

### Areas of Improvement

- **Fine-Tuning Open-Source Models**: Future work includes fine-tuning models like sharded LLaMA for potentially better performance and customization.
- **Enhanced Conversational Memory**: Implementing more advanced memory mechanisms to handle longer and more complex conversations.
- **Improved Source Integration**: Incorporating more diverse and comprehensive sources for a richer knowledge base.
- **Improved Frontend**: The frontend can be improved with respect to UI and dislays etc. time limits did not permit it to be of dynamic nature .

## Deployment Plan and Scaling Approaches

### Deployment Plan

The current deployment is hosted on Huggingface Spaces, providing an accessible and easy-to-use interface. The link to the deployed project is: <https://huggingface.co/spaces/Sambhavnoobcoder/Lecture-Notes-Chatbot>

### Scaling Approaches

As the number of lectures and papers grows, the system can be scaled by:
1. **Incremental Indexing**:
   - Update the FAISS index incrementally as new lecture notes and papers are added, without re-indexing the entire dataset.
   
2. **Distributed Storage**:
   - Use distributed storage solutions like AWS S3 or Google Cloud Storage to manage a large volume of lecture notes and papers.

3. **Sharding and Load Balancing**:
   - Implement sharding for the FAISS index and use load balancing to distribute the query load efficiently.

4. **Asynchronous Processing**:
   - Employ asynchronous processing to handle multiple user queries concurrently, improving response times.

5. **Model Optimization**:
   - Optimize the embedding and response generation models for faster inference times, potentially using model quantization and other techniques.

