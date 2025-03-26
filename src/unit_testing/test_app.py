from unittest.mock import patch, mock_open
import unittest
import datetime
from src.ragchain.app import log_interaction

class TestApp(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("datetime.datetime")
    def test_log_interaction(self, mock_datetime, mock_file):
        mock_datetime.now.return_value = datetime.datetime(2025, 3, 24, 12, 0, 0)
        mock_datetime.now().strftime.return_value = "2025-03-24 12:00:00"

        user_input = "Hello"
        bot_response = "Hi there!"
        log_interaction(user_input, bot_response)

        expected_log = "[2025-03-24 12:00:00] Bot: Hi there!\n\n"
        
        mock_file().write.assert_called_with(expected_log)

if __name__ == "__main__":
    unittest.main()
