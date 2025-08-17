import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_wikipedia_table(self, url, table_index=0):
        """Scrape table from Wikipedia page"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table', class_='wikitable')
            
            if not tables or table_index >= len(tables):
                raise Exception(f"Could not find table at index {table_index}")
            
            # Convert table to pandas DataFrame
            table = tables[table_index]
            
            # Extract headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text().strip())
            
            # Extract rows
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all(['td', 'th']):
                    text = td.get_text().strip()
                    # Clean up text (remove references, etc.)
                    text = text.replace('[citation needed]', '')
                    text = text.split('[')[0]  # Remove reference markers
                    row.append(text)
                if row:
                    rows.append(row)
            
            # Create DataFrame
            if headers and rows:
                # Ensure all rows have same length as headers
                max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
                headers.extend([''] * (max_cols - len(headers)))
                
                for i, row in enumerate(rows):
                    rows[i].extend([''] * (max_cols - len(row)))
                
                df = pd.DataFrame(rows, columns=headers[:max_cols])
                return df
            else:
                raise Exception("Could not extract table data")
                
        except Exception as e:
            logger.error(f"Error scraping Wikipedia table: {str(e)}")
            raise Exception(f"Failed to scrape Wikipedia table: {str(e)}")
    
    def get_website_text_content(self, url):
        """Get main text content from website"""
        try:
            import trafilatura
            
            # Download and extract content
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                raise Exception("Could not download webpage")
                
            text = trafilatura.extract(downloaded)
            if text is None:
                raise Exception("Could not extract text content")
                
            return text
            
        except ImportError:
            # Fallback to simple BeautifulSoup extraction
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
                
            except Exception as e:
                logger.error(f"Error with fallback scraping: {str(e)}")
                raise Exception(f"Failed to scrape website: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error scraping website: {str(e)}")
            raise Exception(f"Failed to scrape website: {str(e)}")
    
    def scrape_with_retry(self, url, max_retries=3, delay=1):
        """Scrape with retry mechanism"""
        for attempt in range(max_retries):
            try:
                return self.get_website_text_content(url)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Scraping attempt {attempt + 1} failed: {str(e)}, retrying...")
                    time.sleep(delay)
                else:
                    raise e
