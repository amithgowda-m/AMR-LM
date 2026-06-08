import urllib.request
import tarfile
import os

url = "https://card.mcmaster.ca/latest/data"
file_path = "card-data.tar.bz2"
extract_dir = os.path.join("data", "raw", "card")

print("Downloading CARD data...")
urllib.request.urlretrieve(url, file_path)

print("Extracting CARD data...")
os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(file_path, "r:bz2") as tar:
    tar.extractall(path=extract_dir)

print("CARD Data downloaded and extracted successfully.")
