# 🎵 Resonance AI - A RAG Chatbot for Music Knowledge Retrieval

## 💫 Project Overview
Resonance AI is a Retrieval-Augmented Generation (RAG) chatbot designed to provide insightful answers to queries related to musicians, music genres, and instruments across different eras and regions. The chatbot leverages Gemini Pro for response generation, combined with a well-structured retrieval system to fetch relevant context before answering user queries.

### **Key Features:**
- **RAG-based Architecture**: Enhances chatbot responses by retrieving factual information before generating answers.
- **Gemini Pro Integration**: Uses Google’s Gemini Pro to generate insightful and contextually rich responses.
- **Streamlit UI**: Provides an intuitive interface for users to interact with the chatbot.
- **Modular Pipeline**: Separates ingestion, retrieval, and generation phases for scalability.
- **Data Sources**: Includes scraped data from various music-related sources covering genres, instruments, and musicians.

## 🎯 Objectives
The main objectives of Resonance AI are:
- To serve as a knowledge hub for music history, genres, and instruments.
- To provide accurate and context-aware answers by leveraging retrieval-augmented generation.
- To enable users to explore music trends across different time periods and regions.
- To integrate an interactive and engaging chatbot interface using Streamlit.

## 🏗️ Architecture Diagram
1. **User Query**: A user inputs a question in the chatbot UI.
2. **Retrieval Phase**: The system searches for relevant information from the vector database (FAISS/Weaviate).
3. **Augmented Context**: The retrieved documents are passed along with the query to Gemini Pro.
4. **Response Generation**: Gemini processes the input and provides a structured response.
5. **User Interaction**: The chatbot displays the response, with optional citations from the retrieved sources.

## 💡 Technologies Used
- **Python** – Primary language for chatbot development.
- **Gemini Pro** – For AI-powered response generation.
- **Streamlit** – For building the chatbot’s user interface.
- **FAISS/Weaviate** – For efficient vector-based retrieval.
- **Scrapy/BeautifulSoup** – For data scraping of musical information.
- **Flask** – API backend to manage requests and responses.
- **LangChain** – For constructing retrieval and generation pipelines.
- **Whisper AI (Future Integration)** – For voice-based query support.


## 📊 Evaluation Framework
Resonance AI’s performance is evaluated using multiple NLP metrics to ensure high-quality and contextually relevant responses.

### **Evaluation Methods:**
- **Exact Match**: Checks if the generated response exactly matches the expected answer.
- **Semantic Similarity**: Computes the cosine similarity between expected and generated answers using a sentence embedding model.
- **ROUGE Scores**: Measures textual overlap using ROUGE-1 and ROUGE-L.
- **BLEU Score**: Evaluates n-gram overlap between expected and generated responses.
- **METEOR Score**: Considers synonym matching and stemming for enhanced evaluation.
- **BERTScore**: Uses contextual embeddings to measure token-level similarity.
- **F1 Score**: Computes precision and recall balance based on common word occurrences.
- **Retrieval Precision**: Assesses the accuracy of retrieved documents in supporting the final answer.

## 🔥 Future Enhancements
- **Slack Integration**: Bringing Resonance AI to Slack for seamless interactions in workspaces.
- **Summarization Chains**: Implementing OpenAI’s GPT-4-turbo or LangChain’s summarization chains for concise responses.
- **Voice Query Support**: Using Whisper AI to process voice queries.
- **Multimodal Capabilities**: Extending Gemini’s capabilities to process images of musical notations or instruments.
- **Enhanced Data Sources**: Integration with Wikipedia, music databases, and digital archives.

