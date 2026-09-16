import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "Kazenowoko/telos"
FILE_PATH = "data/python_corpus_5b.bin"

api = HfApi(token=TOKEN)

file_size_gb = Path(FILE_PATH).stat().st_size / (1024**3)
print(f"Starting upload of {FILE_PATH} ({file_size_gb:.2f} GB) to Hugging Face '{REPO_ID}'...")
start_time = time.time()

api.upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo="data/python_corpus_5b.bin",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="add: 5.0B token Python binary corpus (uint16)"
)

elapsed = time.time() - start_time
print(f"✓ Upload complete in {elapsed/60:.2f} minutes ({file_size_gb / (elapsed/60):.2f} GB/min)!")
