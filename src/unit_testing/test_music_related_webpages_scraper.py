import unittest
from unittest.mock import patch, MagicMock, mock_open
from bs4 import BeautifulSoup
import os
from src.scrapers.music_related_webpages_scraper import MusicScraper

class TestMusicScraper(unittest.TestCase):
    
    @patch("os.makedirs")
    def test_initialization(self, mock_makedirs):
        scraper = MusicScraper("http://example.com", {"id": "test-id"}, output_file="test.txt")
        self.assertEqual(scraper.base_url, "http://example.com")
        self.assertEqual(scraper.content_identifier, {"id": "test-id"})
        self.assertTrue(scraper.output_file.endswith("test.txt"))
        mock_makedirs.assert_called_once_with(scraper.output_dir, exist_ok=True)
    
    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_extract_data_success(self, mock_file, mock_requests):
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.status_code = 200  # Set a valid status code
        mock_response.text = "<div id='test-id'>Sample Text</div>"
        mock_requests.return_value = mock_response

        # Initialize the scraper and call extract_data
        scraper = MusicScraper("http://example.com", {"id": "test-id"})
        scraper.soup = BeautifulSoup(mock_response.text, 'html.parser')
        scraper.extract_data()

        # Assert that the correct data was written to the file
        mock_file().write.assert_called_once_with("Sample Text")

    @patch("requests.get")
    def test_extract_data_no_content(self, mock_requests):
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.status_code = 200  # Set a valid status code
        mock_response.text = "<div>No relevant content</div>"
        mock_requests.return_value = mock_response

        # Initialize the scraper and call extract_data
        scraper = MusicScraper("http://example.com", {"id": "non-existent"})
        scraper.soup = BeautifulSoup(mock_response.text, 'html.parser')

        # Assert that an exception is raised when no content is found
        with self.assertRaises(Exception) as context:
            scraper.extract_data()

        self.assertEqual(str(context.exception), "No content found in the specified div.")

    @patch("selenium.webdriver.Chrome")
    def test_selenium_initialization(self, mock_chrome):
        mock_driver = MagicMock()
        mock_driver.page_source = "<div id='test-id'>Sample Text</div>"
        mock_chrome.return_value = mock_driver
        
        scraper = MusicScraper("http://example.com", {"id": "test-id"}, use_selenium=True)
        self.assertIsNotNone(scraper.soup)
        mock_driver.get.assert_called_once_with("http://example.com")
        mock_driver.quit.assert_called_once()

if __name__ == "__main__":
    unittest.main()
