import unittest
from unittest.mock import patch, MagicMock
import os
from src.ragchain.configure import ConfigLoader, PineconeClient

class TestConfigLoader(unittest.TestCase):
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_google_key", "PINECONE_API_KEY": "test_pinecone_key"})
    @patch("google.generativeai.configure")
    def test_config_loader(self, mock_genai_configure):
        config = ConfigLoader()
        self.assertEqual(config.google_api_key, "test_google_key")
        self.assertEqual(config.pinecone_api_key, "test_pinecone_key")
        mock_genai_configure.assert_called_once_with(api_key="test_google_key")

class TestPineconeClient(unittest.TestCase):
    @patch("src.ragchain.configure.Pinecone")  # Corrected patching
    def test_pinecone_client_initialization(self, mock_pinecone):
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        
        client = PineconeClient(api_key="test_pinecone_key")
        
        mock_pinecone.assert_called_once_with(api_key="test_pinecone_key")  # Now this will pass
        self.assertEqual(client.api_key, "test_pinecone_key")
        self.assertEqual(client.client, mock_pinecone_instance)
        self.assertEqual(client.spec.cloud, "aws")
        self.assertEqual(client.spec.region, "us-east-1")

if __name__ == "__main__":
    unittest.main()
