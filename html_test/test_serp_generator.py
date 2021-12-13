from bs4 import BeautifulSoup


def test_html_file(dirpath, fname):
    config = fname[:4]
    seqeunce = []
    for i in config:
        seqeunce.append(i)
    with open(dirpath+'\\'+fname) as fp:
        soup = BeautifulSoup(fp, 'html.parser')
        mydivs = soup.find_all("div", {"class": "stylelistrow"})
        for div in mydivs:
            if (div["class"] == "searchresult"):
                print
                div