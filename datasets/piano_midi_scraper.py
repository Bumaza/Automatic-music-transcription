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
        elif ':' in td.text:
            m, s = map(int, td.text.split(':'))
            yield (m, s)


def download_midi(url_sufix):
    print('Downloading... ', url_prefix + url_sufix)
    wget.download(url_prefix + url_sufix,  'data/midi/' + url_sufix.split('/')[-1])


def scrape_midi_files():
    midi_files = 'midi_files.htm'
    total_duration = 0
    for composer in scrape_page(url_prefix + midi_files):
        for part in scrape_page(url_prefix + composer):
            if len(part) == 2:
                total_duration += (60 * part[0] + part[-1])
            elif part.endswith('.mid'):
                download_midi(part)
    print('Total duration: {0}h:{1}min'.format(total_duration // 3600, (total_duration % 3600) // 60))


if __name__ == '__main__':
    scrape_midi_files()

