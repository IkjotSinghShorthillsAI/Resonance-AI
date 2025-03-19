import os
from dotenv import load_dotenv
from src.ragchain.vector_embeddings import PineconeIndexManager, EmbeddingModel, DocumentProcessor
from src.ragchain.rag_pipeline import RAGPipeline
from src.ragchain.configure import ConfigLoader, PineconeClient
import streamlit as st
import datetime

# Load environment variables
load_dotenv()

# Load configuration
config = ConfigLoader()
pinecone_client = PineconeClient(api_key=config.pinecone_api_key)
pinecone_manager = PineconeIndexManager(pinecone_client=pinecone_client)
retriever = pinecone_manager.get_retriever()
  
# Initialize the RAG pipeline
st.title('Resonance AI ')
st.markdown("Welcome to Resonance AI! Ask me anything about music genres and instruments, and I'll do my best to answer.")

# Create sidebar to adjust parameters
st.sidebar.title('Model Parameters')
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider('Max Tokens', min_value=1, max_value=4096, value=256)
model = st.sidebar.selectbox("Choose Model", ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"])

rag_pipeline = RAGPipeline(retriever, model=model, max_tokens=max_tokens, temperature=temperature)

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def log_interaction(user_input, bot_response):
    """Logs user input and bot response to a file."""
    with open("rag_chat_log.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] User: {user_input}\n")
        log_file.write(f"[{timestamp}] Bot: {bot_response}\n\n")

# Display previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input and response generation
if prompt := st.chat_input("Enter your query"):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Get response from RAG pipeline
    with st.chat_message('assistant'):
        response = rag_pipeline.answer_question(prompt)
        response_text = response['answer']  # Extract response text
        st.markdown(response_text)
    
    st.session_state['messages'].append({"role": "assistant", "content": response_text})
    
    # Log interaction
    log_interaction(prompt, response_text)

