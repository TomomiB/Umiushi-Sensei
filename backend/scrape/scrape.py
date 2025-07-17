import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO

BASE_URL = "https://seaslug.world"
HEADERS = {
    "User-Agent": "MyScraperBot/1.0"
}
DELAY = 2
output_dir = "species_thumbnails_jpg"
os.makedirs(output_dir, exist_ok=True)

def scrape_species_thumbnails(species_url):
    print(f"Fetching species page: {species_url}")
    response = requests.get(species_url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    thumbnails = soup.select('img.card-img-top[src*="https://img.seaslug.world/"]')

    print(f"Found {len(thumbnails)} thumbnail images.")

    for i, img_tag in enumerate(thumbnails, 1):
        img_url = img_tag.get("src")
        if not img_url.startswith("http"):
            img_url = urljoin(BASE_URL, img_url)

        print(f"Downloading image {i}/{len(thumbnails)}: {img_url}")
        img_response = requests.get(img_url, headers=HEADERS)
        img_response.raise_for_status()

        # Convert and crop the image
        image = Image.open(BytesIO(img_response.content)).convert("RGB")
        width, height = image.size
        cropped_image = image.crop((0, 0, width, height - 30))  # Crop bottom 20px

        filename = f"image_{i:03d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cropped_image.save(filepath, "JPEG", quality=90)
        print(f"Saved cropped image: {filepath}")

        time.sleep(DELAY)

if __name__ == "__main__":
    species_path = "species/thecacera_pacifica"  # Update for each species
    full_url = urljoin(BASE_URL, species_path)
    scrape_species_thumbnails(full_url)