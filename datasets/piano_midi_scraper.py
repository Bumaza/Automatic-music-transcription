import wget
import requests
from bs4 import BeautifulSoup

url_prefix = 'http://www.piano-midi.de/'


def check_status(status_code):
    if status_code != 200:
        print('Exit with status code', status_code)
        exit(0)


def scrape_page(url):
    print('Scraping:', url)
    page = requests.get(url)
    check_status(page.status_code)

    html_doc = page.text
    soup = BeautifulSoup(html_doc, 'html.parser')

    for td in soup.find_all('td', class_='midi'):
        link = td.find('a')
        if link is not None:
            yield link.get('href')


def download_midi(url_sufix):
    print('Downloading... ', url_prefix + url_sufix)
    wget.download(url_prefix + url_sufix,  url_sufix.split('/')[-1])


def scrape_midi_files():
    midi_files = 'midi_files.htm'
    for composer in scrape_page(url_prefix + midi_files):
        for part in scrape_page(url_prefix + composer):
            if part.endswith('.mid'):
                download_midi(part)


if __name__ == '__main__':
    scrape_midi_files()

