from bs4 import BeautifulSoup
import requests
import time
import re
import json
import os
from abc import ABC, abstractmethod

class BaseScraper(ABC):
    def __init__(self, base_url):
        self.base_url = base_url
        self.soup = self._initialize_soup()

    def _initialize_soup(self):
        """Fetches the page and initializes BeautifulSoup"""
        response = requests.get(self.base_url)
        if response.status_code != 200:
            raise Exception(f"Failed to load page {self.base_url}, Status Code: {response.status_code}")
        time.sleep(3)  # Prevent rate-limiting issues
        return BeautifulSoup(response.text, 'html.parser')

    @abstractmethod
    def extract_data(self):
        """Subclasses must implement this method"""
        pass