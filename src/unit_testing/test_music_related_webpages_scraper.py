import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
from src.scrapers.music_related_webpages_scraper import MusicScraper

class TestMusicScraper(unittest.TestCase):
    
    @patch('requests.get')
    def test_extract_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <div id='page-content'>
                    <p>Music history spans centuries.</p>
                    <p>Genres evolve over time.</p>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        scraper = MusicScraper("http://example.com", {"id": "page-content"}, output_file="test_output.txt")
        with patch('builtins.open', mock_open()) as mock_file:
            scraper.extract_data()
        
        mock_file.assert_called_once_with("data/singular_websites/test_output.txt", 'w', encoding='utf-8')
        mock_file().write.assert_called_once_with("Music history spans centuries.\nGenres evolve over time.")
    
    @patch('requests.get')
    def test_extract_data_no_content(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><div></div></body></html>"
        mock_get.return_value = mock_response
        
        scraper = MusicScraper("http://example.com", {"id": "nonexistent"}, output_file="test_output.txt")
        with self.assertRaises(Exception) as context:
            scraper.extract_data()
        
        self.assertIn("No content found in the specified div.", str(context.exception))
    
    @patch('os.makedirs', side_effect=os.makedirs)
    def test_directory_creation(self, mock_makedirs):
        scraper = MusicScraper("http://example.com", {"id": "dummy"}, output_dir="custom_data", output_file="output.txt")
        # Ensure we have some dummy soup to bypass content extraction errors.
        scraper.soup = type("DummySoup", (), {"find": lambda self, tag, **kwargs: type("Dummy", (), {"get_text": lambda s, separator, strip: "Dummy Content"})()})()
        scraper.extract_data()
        
        mock_makedirs.assert_called_once_with("custom_data", exist_ok=True)
        # Optionally add assertions to check the file was created or written using additional patching.

if __name__ == '__main__':
    unittest.main()
