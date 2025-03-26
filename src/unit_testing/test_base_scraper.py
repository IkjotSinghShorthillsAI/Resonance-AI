import unittest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from src.scrapers.base_scraper import BaseScraper

class TestBaseScraper(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://example.com"
    
    @patch("requests.get")
    def test_initialize_soup_requests_success(self, mock_get):
        """Test if BeautifulSoup initializes correctly with requests."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_get.return_value = mock_response

        class TestScraper(BaseScraper):
            def extract_data(self):
                pass

        scraper = TestScraper(self.base_url, use_selenium=False)
        self.assertIsInstance(scraper.soup, BeautifulSoup)
        self.assertEqual(scraper.soup.h1.text, "Test")
    
    @patch("requests.get")
    def test_initialize_soup_requests_failure(self, mock_get):
        """Test if an exception is raised when requests fails."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        class TestScraper(BaseScraper):
            def extract_data(self):
                pass
        
        with self.assertRaises(Exception) as context:
            TestScraper(self.base_url, use_selenium=False)
        self.assertIn("Failed to load page", str(context.exception))
    
    @patch("selenium.webdriver.Chrome")
    def test_initialize_soup_selenium(self, mock_chrome):
        """Test if BeautifulSoup initializes correctly with Selenium."""
        mock_driver = MagicMock()
        mock_driver.page_source = "<html><body><h1>Selenium Test</h1></body></html>"
        mock_chrome.return_value = mock_driver
        
        class TestScraper(BaseScraper):
            def extract_data(self):
                pass
        
        scraper = TestScraper(self.base_url, use_selenium=True)
        self.assertIsInstance(scraper.soup, BeautifulSoup)
        self.assertEqual(scraper.soup.h1.text, "Selenium Test")
        mock_driver.get.assert_called_with(self.base_url)
        mock_driver.quit.assert_called_once()
    
    def test_extract_data_abstract_method(self):
        """Test if the abstract method extract_data raises an error when not implemented."""
        with self.assertRaises(TypeError):
            BaseScraper(self.base_url)

if __name__ == "__main__":
    unittest.main()
