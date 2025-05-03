import requests
from bs4 import BeautifulSoup

url = "https://example.com/faqs"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

with open("data/faqs.txt", "w") as f:
    for q, a in zip(soup.select("h3"), soup.select("p")):
        f.write(f"Q: {q.text.strip()}\nA: {a.text.strip()}\n\n")
