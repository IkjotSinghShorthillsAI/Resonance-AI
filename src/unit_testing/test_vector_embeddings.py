import unittest
from unittest.mock import patch, MagicMock
from src.ragchain.vector_embeddings import DocumentProcessor, EmbeddingModel, PineconeIndexManager

class TestDocumentProcessor(unittest.TestCase):
    @patch("src.ragchain.vector_embeddings.DirectoryLoader")
    @patch("src.ragchain.vector_embeddings.TextLoader")  # Ensure correct loader_cls is used
    def test_load_txt_files(self, mock_text_loader, mock_directory_loader):
        mock_loader_instance = mock_directory_loader.return_value
        mock_loader_instance.load.return_value = ["doc1", "doc2"]
        
        processor = DocumentProcessor("/fake/path")
        result = processor.load_txt_files()
        
        mock_directory_loader.assert_called_once_with("/fake/path", glob="*.txt", loader_cls=mock_text_loader)
        self.assertEqual(result, ["doc1", "doc2"])

    @patch("src.ragchain.vector_embeddings.RecursiveCharacterTextSplitter")
    def test_split_text(self, mock_text_splitter):
        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_documents.return_value = ["chunk1", "chunk2"]
        
        result = DocumentProcessor.split_text(["doc1", "doc2"])
        
        mock_text_splitter.assert_called_once_with(chunk_size=2000, chunk_overlap=200)
        self.assertEqual(result, ["chunk1", "chunk2"])

class TestEmbeddingModel(unittest.TestCase):
    @patch("src.ragchain.vector_embeddings.GoogleGenerativeAIEmbeddings")
    def test_get_embeddings(self, mock_embeddings):
        mock_embeddings_instance = mock_embeddings.return_value
        
        result = EmbeddingModel.get_embeddings()
        
        mock_embeddings.assert_called_once_with(model="models/embedding-001")
        self.assertEqual(result, mock_embeddings_instance)

class TestPineconeIndexManager(unittest.TestCase):
    @patch("src.ragchain.vector_embeddings.PineconeVectorStore")
    @patch("src.ragchain.vector_embeddings.ServerlessSpec")
    def test_create_index(self, mock_serverless_spec, mock_pinecone_vector_store):
        mock_pinecone_client = MagicMock()
        mock_serverless_instance = mock_serverless_spec.return_value
        
        pinecone_manager = PineconeIndexManager(mock_pinecone_client)
        pinecone_manager.create_index()
        
        mock_pinecone_client.client.create_index.assert_called_once_with(
            name="music-rag",
            dimension=768,
            metric="cosine",
            spec=mock_serverless_instance
        )
    
    @patch("src.ragchain.vector_embeddings.PineconeVectorStore.from_documents")
    @patch("src.ragchain.vector_embeddings.EmbeddingModel.get_embeddings")
    def test_upsert_documents(self, mock_get_embeddings, mock_from_documents):
        mock_pinecone_client = MagicMock()
        mock_get_embeddings.return_value = MagicMock()
        mock_from_documents.return_value = MagicMock()
        
        pinecone_manager = PineconeIndexManager(mock_pinecone_client)
        result = pinecone_manager.upsert_documents(["doc1", "doc2"])
        
        mock_from_documents.assert_called_once_with(
            documents=["doc1", "doc2"],
            index_name="music-rag",
            embedding=mock_get_embeddings.return_value
        )
        self.assertEqual(result, mock_from_documents.return_value)
    
    @patch("src.ragchain.vector_embeddings.PineconeVectorStore.from_existing_index")
    @patch("src.ragchain.vector_embeddings.EmbeddingModel.get_embeddings")
    def test_get_retriever(self, mock_get_embeddings, mock_from_existing_index):
        mock_get_embeddings.return_value = MagicMock()
        mock_vector_store_instance = mock_from_existing_index.return_value
        mock_vector_store_instance.as_retriever.return_value = "retriever_instance"
        
        pinecone_manager = PineconeIndexManager(MagicMock())
        result = pinecone_manager.get_retriever()
        
        mock_from_existing_index.assert_called_once_with(
            index_name="music-rag",
            embedding=mock_get_embeddings.return_value
        )
        self.assertEqual(result, "retriever_instance")

if __name__ == "__main__":
    unittest.main()
