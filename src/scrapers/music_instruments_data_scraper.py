from bs4 import BeautifulSoup
import requests
import os
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.scrapers.base_scraper import BaseScraper  # Assuming BaseScraper is in base_scraper.py

class WikipediaScraper(BaseScraper):
    def __init__(self, base_url, output_dir="data/instrument_data"):
        print("Initializing WikipediaScraper...")
        super().__init__(base_url)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def extract_links(self):
        """Extracts links from the Wikipedia page."""
        print("Extracting links from the base page...")
        headings = self.soup.find_all('div', class_='mw-heading mw-heading3')
        links_list = []
        
        for heading in headings:
            next_ul = heading.find_next_sibling('ul')
            if next_ul:
                links = next_ul.find_all('a', href=True)
                for link in links:
                    links_list.append(f"https://en.wikipedia.org{link['href']}")
        
        print(f"Extracted {len(links_list)} links.")
        return links_list
    
    def extract_data(self):
        """Extracts data from each linked page and saves the text."""
        print("Starting data extraction...")
        links = self.extract_links()
        
        for link in links:
            print(f"Fetching data from {link}...")
            try:
                response = self.session.get(link, headers=self.headers)
                if response.status_code != 200:
                    print(f"Failed to retrieve {link} with status code {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                filename = os.path.join(self.output_dir, link.split('/')[-1] + ".txt")
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                
                print(f"Saved content to {filename}")
                time.sleep(random.uniform(3, 7))  # Random delay to avoid detection
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {link}: {e}")

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_period_instruments"
    scraper = WikipediaScraper(url)
    scraper.extract_data()
