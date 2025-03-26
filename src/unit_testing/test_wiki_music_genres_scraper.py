import unittest
from unittest.mock import patch, MagicMock
import os
from src.scrapers.wiki_music_genres_scraper import WikipediaMusicScraper

class TestWikipediaMusicScraper(unittest.TestCase):
    
    def setUp(self):
        self.scraper = WikipediaMusicScraper("https://en.wikipedia.org/wiki/List_of_music_genres_and_styles")

    @patch("src.scrapers.base_scraper.requests.get")
    def test_extract_links(self, mock_get):
        """Test link extraction from Wikipedia page."""
        mock_html = """
        <html>
            <body>
                <td class="sidebar-content hlist">
                    <a href="/wiki/Rock_music">Rock music</a>
                    <a href="/wiki/Blues">Blues</a>
                </td>
                <div class="hatnote navigation-not-searchable">
                    <a href="/wiki/Jazz">Jazz</a>
                </div>
            </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html
        mock_get.return_value = mock_response

        scraper = WikipediaMusicScraper("https://en.wikipedia.org/wiki/List_of_music_genres_and_styles")
        links = scraper.extract_links()

        expected_links = [
            "https://en.wikipedia.org/wiki/Rock_music",
            "https://en.wikipedia.org/wiki/Blues",
            "https://en.wikipedia.org/wiki/Jazz"
        ]

        print("Extracted links:", links)  # Debugging output

        self.maxDiff = None  # Show full diff in failure
        self.assertEqual(links, expected_links)
    @patch('requests.get')
    def test_scrape_genre_page_success(self, mock_get):
        """Test successful page scraping."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<div id='mw-content-text'><p>Genre description.</p></div>"
        mock_get.return_value = mock_response
        
        text = self.scraper.scrape_genre_page("https://en.wikipedia.org/wiki/Rock_music")
        self.assertEqual(text, "Genre description.")
    
    @patch('requests.get')
    def test_scrape_genre_page_failure(self, mock_get):
        """Test handling of failed page fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        text = self.scraper.scrape_genre_page("https://en.wikipedia.org/wiki/Non_existent")
        self.assertIsNone(text)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_text_to_file(self, mock_open):
        """Test saving extracted text to a file."""
        self.scraper.save_text_to_file("Sample text", "test_file.txt")
        mock_open.assert_called_with("test_file.txt", 'w', encoding='utf-8')
        mock_open().write.assert_called_once_with("Sample text")
    
    @patch.object(WikipediaMusicScraper, 'extract_data', return_value=["https://en.wikipedia.org/wiki/Rock_music"])
    @patch.object(WikipediaMusicScraper, 'scrape_genre_page', return_value="Sample genre text")
    @patch.object(WikipediaMusicScraper, 'save_text_to_file')
    def test_run(self, mock_save, mock_scrape, mock_extract):
        """Test full scraping process."""
        with patch('os.makedirs') as mock_makedirs:
            self.scraper.run(output_dir="test_data")
            mock_makedirs.assert_called_with("test_data", exist_ok=True)
            mock_extract.assert_called_once()
            mock_scrape.assert_called_once_with("https://en.wikipedia.org/wiki/Rock_music")
            mock_save.assert_called_once_with("Sample genre text", "test_data/Rock music.txt")

if __name__ == "__main__":
    unittest.main()
