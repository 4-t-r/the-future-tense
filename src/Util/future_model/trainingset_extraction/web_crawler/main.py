import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import csv
import numpy as np
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


class Crawler:

    def __init__(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.future_statement_list = ['might',  "could", "would", "should", "may", "'d", "can", "shall", "musk", "mars", "nasa"]
        self.no_statement_list = ['will', 'would', 'going to', "'ll"]
        self.future_regex = [r"(\d+(?:-\d+)?\+?)\s*(years?)", r"(\d+(?:-\d+)?\+?)\s*(months?)", r"(\d+(?:-\d+)?\+?)\s*(weeks?)"]

    def download_url(self, url):
        return requests.get(url).text

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            self.get_blog_post(url)
            self.add_url_to_visit(url)

    def get_blog_post(self, url):
        content = self.download_url(url)
        soup = BeautifulSoup(content, "html.parser")

        # extrahiere den inhalt aus dem div mit der klasse textcontent
        texts = soup.find_all(text=True)
        visible_texts = filter(tag_visible, texts)
        text = u" ".join(t.strip() for t in visible_texts)

        # print(text)

        # ersetze sonderzeichen mit leerzeichen
        #for char in '=+-,\n':
            #text = text.replace(char, ' ')

        self.get_future_statements_from_text(text)

    def get_future_statements_from_text(self, text):
        punkt_params = PunktParameters()
        #punkt_params.abbrev_types = set(list('Mr', 'Mrs', 'LLC'))
        tokenizer = PunktSentenceTokenizer()
        tokens = tokenizer.tokenize(text)
        with open('potential_future_statements.csv', 'a', encoding='utf-8') as csvfile:
            #csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, delimiter='|', quotechar='')
            for sentence in tokens:
                for substr in self.no_statement_list:
                    if substr in sentence.lower():
                        continue

                #for substr in self.future_statement_list:
                #    if substr in sentence.lower():
                #        print(sentence)
                csvfile.write(sentence + '\n')
                #        break

    def run(self):
        while self.urls_to_visit:
            url = self.urls_to_visit.pop(0)
            logging.info(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)


if __name__ == '__main__':
    Crawler(urls=['https://futurism.com/']).run()
