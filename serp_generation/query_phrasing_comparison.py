import csv
import urllib.request

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

def generate_query(treatment, condition, bias):
    if bias == "positive":
        return treatment + ' effective ' + condition
    elif bias == 'negative':
        return treatment + ' ineffective ' + condition
    else:
        return treatment + ' for ' + condition


def generate_page_url(query):
    return  'https://www.google.com/search?q='+query


def get_query_page(query):
    encoded_query = query.replace(' ','+')
    page_url = generate_page_url(encoded_query)
    req = Request(
       #'https://www.google.com/search?q=ginkgo+biloba+ineffective+tinnitus',
       page_url,
       headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    print(webpage)
    return BeautifulSoup(webpage, 'html.parser')

def get_url(div):
    a = div.find_all('a', href=True)
    for l in a:
        url_str = l.attrs['href']
        end_str = url_str.index('&')
        return url_str[7:end_str]


def get_urls(soup):
    urls = []
    urls_divs = soup.find_all("div", {"class": "egMi0 kCrYT"})
    first_link = soup.find_all("div", {"class": "kCrYT"})
    l1 = get_url(first_link[1])
    if not l1:
        l1 = get_url(first_link[2])
    urls.append(l1)
    for d in urls_divs:
        urls.append(get_url(d))
    return urls


def get_snippets(soup):
    snippets_divs = soup.find_all("div", {"class": "BNeawe s3v9rd AP7Wnd"})
    snippets = []
    for i in range(0, len(snippets_divs)):
        sub_div = snippets_divs[i].find_all("div", {"class": "BNeawe s3v9rd AP7Wnd"})
        if not sub_div:
            continue
       # print('---' + str(i) + '-----------')
       # print(sub_div[0].text)
        snippets.append(sub_div[0].text.strip())
    return snippets


def generate_csv(treatment, condition, bias):
    query = generate_query(treatment, condition, bias)
    soup = get_query_page(query)
    urls = get_urls(soup)
    snippets = get_snippets(soup)
    assert (len(urls) == len(snippets))
    entries = []
    for i in range(0, len(urls)):
        url = urls[i]
        snippet = snippets[i]
        entry = {'url':url,'snippet': snippet}
        entries.append(entry)

    write_dir = 'C:\\research\\falseMedicalClaims\\user study\\snippet study\\annotations\\automatically_generated\\'
    with open(write_dir + query + '.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['url','snippet']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

    #urls = get_urls(soup)




if __name__ == "__main__":
    #generate_csv('ginkgo biloba', 'tinnitus', 'positive')
    generate_csv('omega fatty acids ', 'adhd', 'neutral')


