import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
from time import sleep

SAVE_DIR = "NGA_Images_Fixed"
FAILED_CSV = "failed_images.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

sizes_to_try = [360, 0, 90]

def is_valid_image(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False

def download_image(row, retries=3):
    base_url = row["iiifurl"]
    uuid = row["uuid"]
    file_path = os.path.join(SAVE_DIR, f"{uuid}.jpg")

    # Skip if already downloaded
    if os.path.exists(file_path):
        print(f"Already exists: {uuid}")
        return None

    for attempt in range(retries):
        for size in sizes_to_try:
            url = f"{base_url}/full/full/{size}/default.jpg"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and is_valid_image(r.content):
                    with open(file_path, 'wb') as f:
                        f.write(r.content)
                    print(f"Downloaded: {uuid} at {size}px")
                    return None
                else:
                    print(f"Invalid/empty image: {uuid} at {size}px")
            except requests.exceptions.RequestException as e:
                print(f"Request error: {uuid} at {size}px - {e}")
            sleep(1 + attempt)  # exponential backoff

    print(f"Final fail: {uuid}")
    return {"uuid": uuid, "iiifurl": base_url}

def download_all(csv_path, threads=16):
    df = pd.read_csv(csv_path).dropna(subset=["iiifurl"])
    failed_downloads = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(download_image, row): row for _, row in df.iterrows()}
        for future in as_completed(futures):
            fail = future.result()
            if fail:
                failed_downloads.append(fail)

    if failed_downloads:
        pd.DataFrame(failed_downloads).to_csv(FAILED_CSV, index=False)
        print(f"Some images failed. Logged in {FAILED_CSV}")
    else:
        print("All images downloaded successfully.")

if __name__ == "__main__":
    csv_path = input("Enter path to published_images.csv: ").strip()
    if csv_path == "":
        csv_path = "data\published_images.csv"
    download_all(csv_path)
