import unittest
from unittest.mock import patch, MagicMock
import os
from src.scrapers.wiki_music_genres_scraper import WikipediaMusicScraper

class TestWikipediaMusicScraper(unittest.TestCase):
    
    @patch('requests.get')
    def test_extract_links(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <td class='sidebar-content hlist'>
                    <a href="/wiki/Rock_music">Rock</a>
                    <a href="/wiki/Jazz">Jazz</a>
                </td>
                <div class='hatnote navigation-not-searchable'>
                    <a href="/wiki/Blues">Blues</a>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        scraper = WikipediaMusicScraper("https://en.wikipedia.org/wiki/List_of_music_genres_and_styles")
        links = scraper.extract_links()
        
        self.assertEqual(len(links), 3)
        self.assertIn("https://en.wikipedia.org/wiki/Rock_music", links)
        self.assertIn("https://en.wikipedia.org/wiki/Jazz", links)
        self.assertIn("https://en.wikipedia.org/wiki/Blues", links)
    
    @patch('requests.get')
    def test_scrape_genre_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <div id='mw-content-text'>
                    <p>Rock music is a genre that evolved in the 20th century.</p>
                    <p>It has various subgenres.</p>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        scraper = WikipediaMusicScraper("https://en.wikipedia.org/wiki/List_of_music_genres_and_styles")
        text = scraper.scrape_genre_page("https://en.wikipedia.org/wiki/Rock_music")
        
        self.assertIsNotNone(text)
        self.assertIn("Rock music is a genre that evolved in the 20th century.", text)
        self.assertIn("It has various subgenres.", text)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_text_to_file(self, mock_open):
        scraper = WikipediaMusicScraper("https://en.wikipedia.org/wiki/List_of_music_genres_and_styles")
        scraper.save_text_to_file("Test Content", "test.txt")
        
        mock_open.assert_called_once_with("test.txt", 'w', encoding='utf-8')
        mock_open().write.assert_called_once_with("Test Content")
    
    @patch('os.makedirs')
    @patch('src.scrapers.wiki_music_genres_scraper.WikipediaMusicScraper.extract_data', return_value=["https://en.wikipedia.org/wiki/Rock_music"])
    @patch('src.scrapers.wiki_music_genres_scraper.WikipediaMusicScraper.scrape_genre_page', return_value="Rock music is awesome!")
    @patch('src.scrapers.wiki_music_genres_scraper.WikipediaMusicScraper.save_text_to_file')
    def test_run(self, mock_save, mock_scrape, mock_extract, mock_makedirs):
        scraper = WikipediaMusicScraper("https://en.wikipedia.org/wiki/List_of_music_genres_and_styles")
        scraper.run(output_dir="test_data")
        
        mock_makedirs.assert_called_once_with("test_data", exist_ok=True)
        mock_extract.assert_called_once()
        mock_scrape.assert_called_once_with("https://en.wikipedia.org/wiki/Rock_music")
        mock_save.assert_called_once_with("Rock music is awesome!", "test_data/Rock music.txt")

if __name__ == '__main__':
    unittest.main()
