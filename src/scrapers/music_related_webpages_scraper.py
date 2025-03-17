from base_scraper import BaseScraper
import os
class MusicScraper(BaseScraper):
    def __init__(self, base_url, content_identifier, output_dir="data/singular_websites", output_file="output.txt", use_selenium=False):
        super().__init__(base_url, use_selenium)
        self.content_identifier = content_identifier
        self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, output_file)
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists

    def extract_data(self):
        """Extracts text from the specified div and saves it to a file"""
        content_div = self.soup.find('div', **self.content_identifier)
        if not content_div:
            raise Exception("No content found in the specified div.")
        
        text = content_div.get_text(separator="\n", strip=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Data saved to {self.output_file}")

# Example usage
if __name__ == "__main__":
    scraper1 = MusicScraper("http://lis506-project.wikidot.com/wiki:music", {"id": "page-content"}, output_file="decades.txt")
    scraper1.extract_data()
    
    scraper2 = MusicScraper("https://www.unchainedmusic.io/blog-posts/top-music-genres-in-order-the-most-popular-genres-worldwide", {"class_": "rich-text-block w-richtext"}, output_file="unchainedmusic_top_music_genres.txt")
    scraper2.extract_data()
    
    scraper3 = MusicScraper("https://blog.delivermytune.com/famous-musicians-all-over-india/", {"class_": "entry-content"}, output_file="famous_musicians_all_over_india.txt")
    scraper3.extract_data()
    
    scraper4 = MusicScraper("https://acousticmusic.org/research/history/timeline-of-musical-styles-guitar-history/", {"class_": "et_pb_section et_pb_section_1 et_section_regular"}, output_file="music_instruments.txt", use_selenium=True)
    scraper4.extract_data()
