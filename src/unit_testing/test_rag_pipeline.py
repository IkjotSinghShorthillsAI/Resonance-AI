import unittest
from unittest.mock import patch, MagicMock
from src.ragchain.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    @patch("src.ragchain.rag_pipeline.ChatGoogleGenerativeAI")
    @patch("src.ragchain.rag_pipeline.create_stuff_documents_chain")
    @patch("src.ragchain.rag_pipeline.create_retrieval_chain")
    def test_rag_pipeline_initialization(self, mock_retrieval_chain, mock_stuff_chain, mock_llm):
        mock_retriever = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_stuff_chain.return_value = "mock_question_answer_chain"
        mock_retrieval_chain.return_value = "mock_rag_chain"
        
        rag_pipeline = RAGPipeline(mock_retriever)
        
        mock_llm.assert_called_once_with(model="gemini-2.0-flash", max_tokens=256, temperature=0.7)
        mock_stuff_chain.assert_called_once_with(mock_llm_instance, rag_pipeline.prompt)
        mock_retrieval_chain.assert_called_once_with(mock_retriever, "mock_question_answer_chain")
        
        self.assertEqual(rag_pipeline.rag_chain, "mock_rag_chain")
    
    @patch("src.ragchain.rag_pipeline.ChatGoogleGenerativeAI")
    @patch("src.ragchain.rag_pipeline.create_stuff_documents_chain")
    @patch("src.ragchain.rag_pipeline.create_retrieval_chain")
    def test_rag_pipeline_answer_question(self, mock_retrieval_chain, mock_stuff_chain, mock_llm):
        mock_retriever = MagicMock()
        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = {"answer": "Mock Answer"}
        
        mock_retrieval_chain.return_value = mock_rag_chain
        rag_pipeline = RAGPipeline(mock_retriever)
        rag_pipeline.rag_chain = mock_rag_chain
        
        response = rag_pipeline.answer_question("Test query")
        
        mock_rag_chain.invoke.assert_called_once_with({"input": "Test query"})
        self.assertEqual(response["answer"], "Mock Answer")

if __name__ == "__main__":
    unittest.main()
