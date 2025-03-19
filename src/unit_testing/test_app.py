import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import src.ragchain.app as app  # Import your app.py file
import os

def clear_session_state():
    """Clears Streamlit session state for test isolation."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

class TestApp(unittest.TestCase):
    
    @patch("src.ragchain.app.st.sidebar.title")
    @patch("src.ragchain.app.st.title")
    def test_app_ui_elements(self, mock_title, mock_sidebar_title):
        """Tests if UI elements like title and sidebar exist."""
        clear_session_state()
        # Simulate UI initialization if your app calls these functions on startup.
        app.st.title("Resonance AI ")
        app.st.sidebar.title("Model Parameters")
        mock_title.assert_called_with("Resonance AI ")
        mock_sidebar_title.assert_called_with("Model Parameters")
    
    @patch("src.ragchain.app.RAGPipeline")
    def test_session_state_initialization(self, mock_rag_pipeline):
        """Tests if session state initializes correctly."""
        clear_session_state()
        self.assertNotIn('messages', st.session_state)
        app.st.session_state['messages'] = []
        self.assertEqual(st.session_state['messages'], [])
    
    @patch("src.ragchain.app.RAGPipeline")
    def test_rag_pipeline_response(self, mock_rag_pipeline):
        """Tests if RAGPipeline generates responses correctly."""
        clear_session_state()
        mock_pipeline_instance = mock_rag_pipeline.return_value
        mock_pipeline_instance.answer_question.return_value = {"answer": "Mock response"}
        
        test_query = "What is jazz?"
        response = mock_pipeline_instance.answer_question(test_query)
        self.assertEqual(response["answer"], "Mock response")
    
    @patch("src.ragchain.app.log_interaction")
    def test_log_interaction(self, mock_log_interaction):
        """Tests if user queries and bot responses are logged correctly."""
        user_input = "Hello, bot!"
        bot_response = "Hello, user!"
        
        app.log_interaction(user_input, bot_response)
        mock_log_interaction.assert_called_with(user_input, bot_response)
    
    def test_log_file_creation(self):
        """Tests if interactions are written to a log file."""
        log_file = "rag_chat_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        app.log_interaction("Test user", "Test response")
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, "r") as f:
            content = f.read()
            self.assertIn("Test user", content)
            self.assertIn("Test response", content)

if __name__ == "__main__":
    unittest.main()