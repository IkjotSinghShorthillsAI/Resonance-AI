from bs4 import BeautifulSoup
import requests
import time
import os
from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class BaseScraper(ABC):
    def __init__(self, base_url, use_selenium=False):
        self.base_url = base_url
        self.use_selenium = use_selenium
        self.soup = self._initialize_soup()

    def _initialize_soup(self):
        """Fetches the page and initializes BeautifulSoup, optionally using Selenium"""
        if self.use_selenium:
            options = Options()
            options.add_argument("--headless")  # Run Chrome in headless mode
            driver = webdriver.Chrome(service=Service(), options=options)
            driver.get(self.base_url)
            time.sleep(3)  # Allow time for dynamic content to load
            page_source = driver.page_source
            driver.quit()
        else:
            response = requests.get(self.base_url)
            if response.status_code != 200:
                raise Exception(f"Failed to load page {self.base_url}, Status Code: {response.status_code}")
            time.sleep(3)  # Prevent rate-limiting issues
            page_source = response.text
        
        return BeautifulSoup(page_source, 'html.parser')

    @abstractmethod
    def extract_data(self):
        """Subclasses must implement this method"""
        pass
