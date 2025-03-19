import unittest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
import requests
from src.scrapers.base_scraper import BaseScraper

class TestBaseScraper(unittest.TestCase):
    
    @patch('requests.get')
    def test_initialize_soup_requests(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>Test Page</p></body></html>"
        mock_get.return_value = mock_response
        
        class TestScraper(BaseScraper):
            def extract_data(self):
                pass
        
        scraper = TestScraper("http://example.com", use_selenium=False)
        self.assertIsInstance(scraper.soup, BeautifulSoup)
        self.assertEqual(scraper.soup.p.text, "Test Page")
    
    @patch('selenium.webdriver.Chrome')
    def test_initialize_soup_selenium(self, mock_chrome):
        mock_driver = MagicMock()
        mock_driver.page_source = "<html><body><p>Test Selenium Page</p></body></html>"
        mock_chrome.return_value = mock_driver
        
        class TestScraper(BaseScraper):
            def extract_data(self):
                pass
        
        scraper = TestScraper("http://example.com", use_selenium=True)
        self.assertIsInstance(scraper.soup, BeautifulSoup)
        self.assertEqual(scraper.soup.p.text, "Test Selenium Page")
        
        mock_driver.get.assert_called_once_with("http://example.com")
        mock_driver.quit.assert_called_once()
    
    @patch('requests.get')
    def test_initialize_soup_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        class TestScraper(BaseScraper):
            def extract_data(self):
                pass
        
        with self.assertRaises(Exception) as context:
            TestScraper("http://example.com", use_selenium=False)
        
        self.assertIn("Failed to load page", str(context.exception))

if __name__ == '__main__':
    unittest.main()
