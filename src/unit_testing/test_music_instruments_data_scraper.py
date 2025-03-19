import unittest
import requests
from unittest.mock import patch, MagicMock, mock_open
import os
from src.scrapers.music_instruments_data_scraper import WikipediaScraper

class TestWikipediaScraper(unittest.TestCase):
    
    @patch('requests.get')
    def test_extract_links(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <div class='mw-heading mw-heading3'></div>
                <ul>
                    <li><a href='/wiki/Instrument1'>Instrument1</a></li>
                    <li><a href='/wiki/Instrument2'>Instrument2</a></li>
                </ul>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        scraper = WikipediaScraper("https://en.wikipedia.org/wiki/List_of_period_instruments")
        links = scraper.extract_links()
        
        expected_links = [
            "https://en.wikipedia.org/wiki/Instrument1",
            "https://en.wikipedia.org/wiki/Instrument2"
        ]
        self.assertEqual(links, expected_links)

    
    @patch('requests.Session.get')
    def test_extract_data(self, mock_session_get):
        mock_link_response = MagicMock()
        mock_link_response.status_code = 200
        mock_link_response.text = "Some instrument details."
        mock_session_get.return_value = mock_link_response
        
        scraper = WikipediaScraper("https://en.wikipedia.org/wiki/List_of_period_instruments")
        scraper.extract_links = MagicMock(return_value=["https://en.wikipedia.org/wiki/Instrument1"])
        
        with patch('builtins.open', mock_open()) as mock_file:
            scraper.extract_data()
        
        mock_file.assert_called_once_with("data/instrument_data/Instrument1.txt", 'w', encoding='utf-8')
        mock_file().write.assert_called_once_with("Some instrument details.")
    
    @patch('requests.Session.get')
    def test_extract_data_request_failure(self, mock_session_get):
        mock_session_get.side_effect = requests.exceptions.RequestException("Request failed")
        
        scraper = WikipediaScraper("https://en.wikipedia.org/wiki/List_of_period_instruments")
        scraper.extract_links = MagicMock(return_value=["https://en.wikipedia.org/wiki/Instrument1"])
        
        with patch('builtins.print') as mock_print:
            scraper.extract_data()
        
        mock_print.assert_any_call("Request failed for https://en.wikipedia.org/wiki/Instrument1: Request failed")

if __name__ == '__main__':
    unittest.main()
