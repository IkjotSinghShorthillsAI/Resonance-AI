import unittest
from unittest.mock import patch, MagicMock
from src.ragchain.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    @patch("src.ragchain.rag_pipeline.ChatGoogleGenerativeAI")
    @patch("src.ragchain.rag_pipeline.create_stuff_documents_chain")
    @patch("src.ragchain.rag_pipeline.create_retrieval_chain")
    def test_pipeline_initialization(self, mock_create_retrieval_chain, mock_create_stuff_documents_chain, mock_chat_model):
        mock_retriever = MagicMock()
        mock_llm_instance = mock_chat_model.return_value
        mock_stuff_chain_instance = mock_create_stuff_documents_chain.return_value
        mock_retrieval_chain_instance = mock_create_retrieval_chain.return_value
        
        pipeline = RAGPipeline(mock_retriever)
        
        mock_chat_model.assert_called_once_with(model="gemini-2.0-flash", max_tokens=256, temperature=0.7)
        mock_create_stuff_documents_chain.assert_called_once_with(mock_llm_instance, pipeline.prompt)
        mock_create_retrieval_chain.assert_called_once_with(mock_retriever, mock_stuff_chain_instance)
        
        self.assertEqual(pipeline.llm, mock_llm_instance)
        self.assertEqual(pipeline.question_answer_chain, mock_stuff_chain_instance)
        self.assertEqual(pipeline.rag_chain, mock_retrieval_chain_instance)

    @patch("src.ragchain.rag_pipeline.ChatGoogleGenerativeAI")
    @patch("src.ragchain.rag_pipeline.create_stuff_documents_chain")
    @patch("src.ragchain.rag_pipeline.create_retrieval_chain")
    def test_answer_question(self, mock_create_retrieval_chain, mock_create_stuff_documents_chain, mock_chat_model):
        mock_retriever = MagicMock()
        mock_retrieval_chain_instance = mock_create_retrieval_chain.return_value
        mock_retrieval_chain_instance.invoke.return_value = {"answer": "Rock, Jazz, and Classical."}
        
        pipeline = RAGPipeline(mock_retriever)
        result = pipeline.answer_question("What are the top 3 genres of music?")
        
        mock_retrieval_chain_instance.invoke.assert_called_once_with({"input": "What are the top 3 genres of music?"})
        self.assertEqual(result["answer"], "Rock, Jazz, and Classical.")

if __name__ == "__main__":
    unittest.main()
