import unittest
from unittest.mock import patch, MagicMock
from src.ragchain.vector_embeddings import DocumentProcessor, EmbeddingModel, PineconeIndexManager

class TestDocumentProcessor(unittest.TestCase):
    @patch("src.ragchain.vector_embeddings.TextLoader")
    @patch("src.ragchain.vector_embeddings.DirectoryLoader")
    def test_load_txt_files(self, mock_directory_loader, mock_text_loader):
        mock_loader_instance = MagicMock()
        mock_directory_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = ["Mock Document"]

        processor = DocumentProcessor("test_path")
        result = processor.load_txt_files()

        mock_directory_loader.assert_called_once_with("test_path", glob="*.txt", loader_cls=mock_text_loader)
        self.assertEqual(result, ["Mock Document"])
    
    @patch("src.ragchain.vector_embeddings.RecursiveCharacterTextSplitter")
    def test_split_text(self, mock_text_splitter):
        mock_splitter_instance = MagicMock()
        mock_text_splitter.return_value = mock_splitter_instance
        mock_splitter_instance.split_documents.return_value = ["Chunk1", "Chunk2"]
        
        result = DocumentProcessor.split_text(["Mock Document"], chunk_size=1000, chunk_overlap=100)
        
        mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=100)
        self.assertEqual(result, ["Chunk1", "Chunk2"])

class TestEmbeddingModel(unittest.TestCase):
    @patch("src.ragchain.vector_embeddings.GoogleGenerativeAIEmbeddings")
    def test_get_embeddings(self, mock_embedding_class):
        mock_instance = MagicMock()
        mock_embedding_class.return_value = mock_instance

        result = EmbeddingModel.get_embeddings()
        
        mock_embedding_class.assert_called_once_with(model="models/embedding-001")
        self.assertEqual(result, mock_instance)

class TestPineconeIndexManager(unittest.TestCase):
    @patch("src.ragchain.vector_embeddings.PineconeVectorStore.from_documents")
    @patch("src.ragchain.vector_embeddings.EmbeddingModel.get_embeddings")
    def test_upsert_documents(self, mock_get_embeddings, mock_from_documents):
        mock_embedding_instance = MagicMock()
        mock_get_embeddings.return_value = mock_embedding_instance

        mock_pinecone_instance = MagicMock()
        mock_from_documents.return_value = mock_pinecone_instance

        pinecone_client = MagicMock()
        manager = PineconeIndexManager(pinecone_client)
        result = manager.upsert_documents(["Mock Document"])

        mock_get_embeddings.assert_called_once()
        mock_from_documents.assert_called_once_with(
            documents=["Mock Document"],
            index_name="music-rag",
            embedding=mock_embedding_instance
        )
        self.assertEqual(result, mock_pinecone_instance)

    @patch("src.ragchain.vector_embeddings.PineconeVectorStore.from_existing_index")
    @patch("src.ragchain.vector_embeddings.EmbeddingModel.get_embeddings")
    def test_get_retriever(self, mock_get_embeddings, mock_from_existing_index):
        mock_embedding_instance = MagicMock()
        mock_get_embeddings.return_value = mock_embedding_instance

        mock_retriever = MagicMock()
        mock_from_existing_index.return_value = mock_retriever

        manager = PineconeIndexManager(MagicMock())
        result = manager.get_retriever()

        mock_get_embeddings.assert_called_once()
        mock_from_existing_index.assert_called_once_with(
            index_name="music-rag",
            embedding=mock_embedding_instance
        )
        self.assertEqual(result, mock_retriever.as_retriever.return_value)

    @patch("src.ragchain.vector_embeddings.ServerlessSpec")
    @patch("src.ragchain.vector_embeddings.PineconeClient")
    def test_create_index(self, mock_pinecone_client, mock_serverless_spec):
        mock_client_instance = MagicMock()
        mock_pinecone_client.return_value = mock_client_instance

        mock_spec_instance = MagicMock()
        mock_serverless_spec.return_value = mock_spec_instance

        manager = PineconeIndexManager(mock_client_instance)
        manager.create_index()

        mock_client_instance.client.create_index.assert_called_once_with(
            name="music-rag",
            dimension=768,
            metric="cosine",
            spec=mock_spec_instance
        )

if __name__ == "__main__":
    unittest.main()
