from bs4 import BeautifulSoup
import requests
import html
import os
import zipfile

f = open("Foram Data Portal.html", "r")
response = f.read()
f.close()

soup = BeautifulSoup(response, 'html.parser')

labels = []
counts = []

for tag in soup.find('div', id='specimens').find_all('div', class_='col'):
    labels.append(tag.find('i').text)
    counts.append(int(tag.find('div', 'overlay').text.split(' ')[0]))

os.makedirs("zip_files", exist_ok=True)
os.makedirs("specimens", exist_ok=True)
for i, label in enumerate(labels):
    zip_filename = os.path.join("zip_files", label + ".zip")
    extract_dir = os.path.join("specimens", label.replace(" ", "_"))
    if os.path.exists(extract_dir) is False:
        print("Downloading {} ({} images)".format(label, counts[i]))
        r = requests.get("http://endlessforams.org/randomizer/download/{}/0".format(html.escape(label)))
        with open(zip_filename, 'wb') as w:
            w.write(r.content)
        print("Extracting to {}".format(extract_dir))
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print("Skipping {} as it already exists".format(label))
