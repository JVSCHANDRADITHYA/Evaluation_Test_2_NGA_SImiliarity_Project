import pandas as pd
import requests
import os

# Load CSV file
data = pd.read_csv(r"F:\NGA_Similarity_Project\data\published_images.csv")  # Update with the correct file path

# Define the output directory (Google Drive path)
output_dir = r"D:\NGA_open_dataset"
os.makedirs(output_dir, exist_ok=True)

# Log file path
log_file = os.path.join(output_dir, "download_log.txt")

# Open the log file in append mode
with open(log_file, "a") as log:
    log.write("Download Log\n")
    log.write("=" * 50 + "\n")

# Loop through iiif URLs and download images
for _, row in data.dropna(subset=["iiifurl"]).iterrows():
    try:
        base_url = row["iiifurl"]
        uuid = row["uuid"]  # Extracting the UUID for naming
        image_url = f"{base_url}/full/full/0/default.jpg"
        file_name = f"{uuid}.jpg"  # Naming the file with its UUID
        file_path = os.path.join(output_dir, file_name)

        response = requests.get(image_url, stream=True)

        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            # Log the download
            with open(log_file, "a") as log:
                log.write(f"Downloaded: {file_name} | URL: {image_url}\n")

            print(f"Downloaded: {file_name}")

        else:
            print(f"Failed: {image_url}")

    except Exception as e:
        print(f"Error downloading {base_url}: {e}")

print("Download complete!")