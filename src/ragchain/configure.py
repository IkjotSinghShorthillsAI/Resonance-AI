import os
import time
from dotenv import load_dotenv
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
import google.generativeai as genai

class ConfigLoader:
    """Loads environment variables and configures API keys."""
    def __init__(self):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        genai.configure(api_key=self.google_api_key)

class PineconeClient:
    """Handles Pinecone connection and serverless configuration."""
    def __init__(self, api_key: str, cloud: str = "aws", region: str = "us-east-1"):
        self.api_key = api_key
        self.client = Pinecone(api_key=self.api_key)
        self.spec = ServerlessSpec(cloud=cloud, region=region)

if __name__ == "__main__":
    # Load configuration
    config = ConfigLoader()
    
    # Initialize Pinecone client
    pinecone_client = PineconeClient(api_key=config.pinecone_api_key)
    
