import os
from src.ragchain.configure import ConfigLoader, PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import ServerlessSpec

class DocumentProcessor:
    """Handles loading and splitting of text documents."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_txt_files(self):
        loader = DirectoryLoader(self.data_path, glob="*.txt", loader_cls=TextLoader)
        return loader.load()

    @staticmethod
    def split_text(extracted_data, chunk_size=2000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(extracted_data)

class EmbeddingModel:
    """Handles embedding model selection and initialization."""
    @staticmethod
    def get_embeddings():
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class PineconeIndexManager:
    """Handles Pinecone index creation and vector storage."""
    def __init__(self, pinecone_client, index_name="music-rag", dimension=768, metric="cosine"):
        self.index_name = index_name
        self.pinecone_client = pinecone_client
        self.dimension = dimension
        self.metric = metric
        self.spec = ServerlessSpec(cloud="aws", region="us-east-1")
        
    def create_index(self):
        self.pinecone_client.client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=self.spec
        )
    
    def upsert_documents(self, documents):
        return PineconeVectorStore.from_documents(
            documents=documents,
            index_name=self.index_name,
            embedding=EmbeddingModel.get_embeddings()
        )

    def get_retriever(self):
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=EmbeddingModel.get_embeddings()
        )
        return docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

if __name__ == "__main__":
    # Load configuration
    config = ConfigLoader()
    pinecone_client = PineconeClient(api_key=config.pinecone_api_key)
    
    folder_paths = [
        "/home/shtlp_0096/Desktop/coding/rag_project/data/singular_websites",
        "/home/shtlp_0096/Desktop/coding/rag_project/data/instrument_data",
        "/home/shtlp_0096/Desktop/coding/rag_project/data/scraped_genres"
    ]
    
    all_text_chunks = []
    for path in folder_paths:
        processor = DocumentProcessor(path)
        extracted_data = processor.load_txt_files()
        text_chunks = processor.split_text(extracted_data)
        print(len(text_chunks))
        all_text_chunks.extend(text_chunks)
    print(len(all_text_chunks))
    
    pinecone_manager = PineconeIndexManager(pinecone_client=pinecone_client)
    pinecone_manager.create_index()
    pinecone_manager.upsert_documents(all_text_chunks)
    print("All documents upserted to Pinecone index.")
