import argparse
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"

# Dataset configuration
CLEARVQA_PATH = DATA_DIR / "clearvqa"
CLEARVQA_IMAGES_URL = "https://huggingface.co/datasets/jian0418/ClearVQA/resolve/main/images.zip"
CLEARVQA_TRAIN_TABLE_URL = "https://huggingface.co/datasets/jian0418/ClearVQA/resolve/main/train_annotated.jsonl"
CLEARVQA_VAL_TABLE_URL = "https://huggingface.co/datasets/jian0418/ClearVQA/resolve/main/val_annotated.jsonl"
CLEARVQA_IMAGE_ZIP_PATH = CLEARVQA_PATH / "images.zip"
CLEARVQA_TRAIN_TABLE_PATH = CLEARVQA_PATH / "train_annotated.jsonl"
CLEARVQA_VAL_TABLE_PATH = CLEARVQA_PATH / "val_annotated.jsonl"

# val_004563

def download_file(file_url: str, download_path: Path):
    # Streaming, so we can iterate over the response.
    response = requests.get(file_url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(download_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

    # try:
    #     response = requests.get(file_url, stream=True)  # Use stream=True for large files
    #     response.raise_for_status()  # Check for HTTP errors

    #     with open(download_path, 'wb') as f:
    #         for chunk in response.iter_content(chunk_size=8192):
    #             f.write(chunk)
    #     print(f"File downloaded successfully to {download_path}")
    #     return True
    # except requests.exceptions.RequestException as e:
    #     print(f"Error downloading file: {e}")
    #     return False

def unzip_file(file_path: Path, extract_dir: Path):
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zf:
        zf.extractall(extract_dir)

def download_clearvqa():
    CLEARVQA_PATH.mkdir(exist_ok=True)
    if CLEARVQA_IMAGE_ZIP_PATH.exists():
        print("ClearVQA images zip already downloaded. Skipping.")
    else:
        download_file(CLEARVQA_IMAGES_URL, CLEARVQA_IMAGE_ZIP_PATH)

    if (CLEARVQA_PATH / "images").exists():
        print("ClearVQA image folder already exists. Skipping extraction.")
    else:
        unzip_file(CLEARVQA_IMAGE_ZIP_PATH, CLEARVQA_PATH)

    if CLEARVQA_TRAIN_TABLE_PATH.exists():
        print("ClearVQA train annotation table already exists. Skipping.")
    else:
        download_file(CLEARVQA_TRAIN_TABLE_URL, CLEARVQA_TRAIN_TABLE_PATH)

    if CLEARVQA_VAL_TABLE_PATH.exists():
        print("ClearVQA val annotation table already exists. Skipping.")
    else:
        download_file(CLEARVQA_VAL_TABLE_URL, CLEARVQA_VAL_TABLE_PATH)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--clearvqa', action='store_true', help='Download ClearVQA dataset')
    args = parser.parse_args()

    print("Starting downloads")
    if args.clearvqa:
        download_clearvqa()


if __name__ == "__main__":
    main()
