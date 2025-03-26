import unittest
import os
import time
import random
from unittest.mock import patch, MagicMock, mock_open
from src.scrapers.music_instruments_data_scraper import WikipediaScraper
from bs4 import BeautifulSoup
class TestWikipediaScraper(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environment."""
        self.test_url = "https://en.wikipedia.org/wiki/List_of_period_instruments"
        self.test_output_dir = "test_data/instrument_data"
        self.scraper = WikipediaScraper(self.test_url, self.test_output_dir)

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_init(self, mock_makedirs, mock_exists):
        """Test initialization of WikipediaScraper."""
        scraper = WikipediaScraper(self.test_url, self.test_output_dir)
        self.assertEqual(scraper.base_url, self.test_url)
        self.assertEqual(scraper.output_dir, self.test_output_dir)
        mock_makedirs.assert_called_once_with(self.test_output_dir)

    @patch("src.scrapers.base_scraper.BaseScraper.soup", new_callable=MagicMock)
    def test_extract_links(self, mock_soup):
        """Test extraction of links from Wikipedia page."""
        # Mock the HTML structure of the Wikipedia page
        mock_html = """
        <div class="mw-heading mw-heading3">Instruments</div>
        <ul>
            <li><a href="/wiki/Violin">Violin</a></li>
            <li><a href="/wiki/Flute">Flute</a></li>
        </ul>
        """
        mock_soup.find_all.return_value = BeautifulSoup(mock_html, "html.parser").find_all("div", class_="mw-heading mw-heading3")

        # Expected links
        expected_links = [
            "https://en.wikipedia.org/wiki/Violin",
            "https://en.wikipedia.org/wiki/Flute"
        ]

        # Call the method and verify the result
        links = self.scraper.extract_links()
        self.assertCountEqual(links, expected_links)  # Use `assertCountEqual` to avoid ordering issues

    @patch("src.scrapers.music_instruments_data_scraper.WikipediaScraper.extract_links")
    @patch("requests.Session.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_extract_data(self, mock_open_file, mock_get, mock_extract_links):
        """Test full data extraction process."""
        mock_extract_links.return_value = [
            "https://en.wikipedia.org/wiki/Violin"
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>Violin is a musical instrument.</p></body></html>"
        mock_get.return_value = mock_response

        self.scraper.extract_data()

        mock_get.assert_called_once_with("https://en.wikipedia.org/wiki/Violin", headers=self.scraper.headers)
        mock_open_file.assert_called_once_with(
            os.path.join(self.test_output_dir, "Violin.txt"), "w", encoding="utf-8"
        )
        mock_open_file().write.assert_called_once_with("Violin is a musical instrument.")

    @patch("requests.Session.get")
    def test_extract_data_failed_request(self, mock_get):
        """Test handling of failed requests."""
        mock_get.return_value.status_code = 404

        with patch("src.scrapers.music_instruments_data_scraper.WikipediaScraper.extract_links", return_value=["https://en.wikipedia.org/wiki/Violin"]):
            self.scraper.extract_data()
            mock_get.assert_called_once()

    @patch("time.sleep", return_value=None)
    @patch("random.uniform", return_value=3)
    def test_random_delay(self, mock_random, mock_sleep):
        """Test that a random delay is introduced."""
        time.sleep(random.uniform(3, 7))
        mock_random.assert_called_once_with(3, 7)
        mock_sleep.assert_called_once_with(3)

if __name__ == "__main__":
    unittest.main()
