import os
import json
import requests
from bs4 import BeautifulSoup
from base_scraper import BaseScraper

class WikipediaMusicScraper(BaseScraper):
    def extract_data(self):
        """Extracts and processes all genre links and their content."""
        print("Extracting genre links from the main Wikipedia page...")
        return self.extract_links()
    
    def extract_links(self):
        """Extracts genre links from the main Wikipedia page."""
        print("Running extract_links method...")
        genre_links = []
        td = self.soup.find('td', class_='sidebar-content hlist')
        if td:
            for link in td.find_all('a', href=True):
                genre_links.append("https://en.wikipedia.org" + link['href'])
        
        div_links = []
        link_divs = self.soup.find_all('div', class_='hatnote navigation-not-searchable')
        for div in link_divs:
            a = div.find('a', href=True)
            if a:
                div_links.append("https://en.wikipedia.org" + a['href'])
        
        print(f"Extracted {len(genre_links) + len(div_links)} links.")
        return genre_links + div_links

    def scrape_genre_page(self, url):
        """Fetches and extracts text content from a genre Wikipedia page."""
        print(f"Scraping content from: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', {'id': 'mw-content-text'})
        if content:
            paragraphs = content.find_all('p')
            text = "\n".join(p.get_text() for p in paragraphs)
            print(f"Extracted text from {url}")
            return text.strip()
        
        print(f"No text found on {url}")
        return None

    def save_text_to_file(self, text, filename):
        """Saves extracted text into a file."""
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Saved file: {filename}")

    def run(self, output_dir="data/scraped_genres"):
        """Runs the full scraping pipeline: extract links, fetch content, save to files."""
        print("Starting scraping process...")
        os.makedirs(output_dir, exist_ok=True)
        links = self.extract_data()
        
        for link in links:
            genre_name = link.split("/")[-1].replace("_", " ")
            print(f"Processing genre: {genre_name}")
            text = self.scrape_genre_page(link)
            if text:
                filename = os.path.join(output_dir, f"{genre_name}.txt")
                self.save_text_to_file(text, filename)
        print("Scraping process completed!")

# Example usage
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_music_genres_and_styles"
    scraper = WikipediaMusicScraper(url)
    scraper.run()