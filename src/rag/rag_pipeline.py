import os
from config import ConfigLoader, PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import ServerlessSpec
from vector_embeddings import DocumentProcessor, EmbeddingModel, PineconeIndexManager

class RAGPipeline:
    """Handles the RAG pipeline for retrieval and question answering."""
    def __init__(self, retriever,model="gemini-2.0-flash",max_tokens=256,temperature=0.7):
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(model=model, max_tokens=max_tokens, temperature=temperature)
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

    def answer_question(self, query: str):
        return self.rag_chain.invoke({"input": query})

if __name__ == "__main__":
    # Load configuration
    config = ConfigLoader()
    pinecone_client = PineconeClient(api_key=config.pinecone_api_key)

    
    pinecone_manager = PineconeIndexManager(pinecone_client=pinecone_client)
    
    retriever = pinecone_manager.get_retriever()
    
    # Initialize and test RAG pipeline
    rag_pipeline = RAGPipeline(retriever)
    print("Ready to answer questions!")
    response = rag_pipeline.answer_question("What are the top 3 genres of music?")
    print(response['answer'])
