from base_scraper import BaseScraper
import os

class MusicScraper(BaseScraper):
    def __init__(self, base_url, output_dir="data/singular_websites", output_file="decades.txt"):
        super().__init__(base_url)
        self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, output_file)
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists

    def extract_data(self):
        """Extracts text from the div with id 'page-content' and saves it to a file"""
        content_div = self.soup.find('div', id='page-content')
        if not content_div:
            raise Exception("No content found in 'page-content' div.")
        
        text = content_div.get_text()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Data saved to {self.output_file}")

# Example usage
if __name__ == "__main__":
    scraper = MusicScraper("http://lis506-project.wikidot.com/wiki:music")
    scraper.extract_data()
