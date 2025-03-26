import unittest
from unittest.mock import patch, MagicMock
import os
from src.ragchain.configure import ConfigLoader, PineconeClient

class TestConfigLoader(unittest.TestCase):
    @patch("src.ragchain.configure.load_dotenv")
    @patch("src.ragchain.configure.os.getenv")
    @patch("src.ragchain.configure.genai.configure")
    def test_config_loader(self, mock_genai_configure, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: "test-google-api-key" if key == "GOOGLE_API_KEY" else "test-pinecone-api-key"
        
        config = ConfigLoader()
        
        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_any_call("GOOGLE_API_KEY")
        mock_getenv.assert_any_call("PINECONE_API_KEY")
        mock_genai_configure.assert_called_once_with(api_key="test-google-api-key")
        
        self.assertEqual(config.google_api_key, "test-google-api-key")
        self.assertEqual(config.pinecone_api_key, "test-pinecone-api-key")
    
    @patch("src.ragchain.configure.os.getenv", side_effect=lambda key: None)
    def test_config_loader_missing_keys(self, mock_getenv):
        with self.assertRaises(TypeError):
            ConfigLoader()

class TestPineconeClient(unittest.TestCase):
    @patch("src.ragchain.configure.Pinecone")
    @patch("src.ragchain.configure.ServerlessSpec")
    def test_pinecone_client(self, mock_serverless_spec, mock_pinecone):
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        mock_serverless_spec.return_value = "mock-spec"

        client = PineconeClient(api_key="test-pinecone-api-key", cloud="gcp", region="us-west-2")
        
        mock_pinecone.assert_called_once_with(api_key="test-pinecone-api-key")
        mock_serverless_spec.assert_called_once_with(cloud="gcp", region="us-west-2")
        
        self.assertEqual(client.api_key, "test-pinecone-api-key")
        self.assertEqual(client.client, mock_pinecone_instance)
        self.assertEqual(client.spec, "mock-spec")
    
    @patch("src.ragchain.configure.Pinecone", side_effect=Exception("Pinecone error"))
    def test_pinecone_client_initialization_failure(self, mock_pinecone):
        with self.assertRaises(Exception) as context:
            PineconeClient(api_key="invalid-key")
        self.assertEqual(str(context.exception), "Pinecone error")
    
    @patch("src.ragchain.configure.Pinecone")
    def test_pinecone_client_default_values(self, mock_pinecone):
        client = PineconeClient(api_key="test-pinecone-api-key")
        self.assertEqual(client.spec.cloud, "aws")
        self.assertEqual(client.spec.region, "us-east-1")

if __name__ == "__main__":
    unittest.main()
