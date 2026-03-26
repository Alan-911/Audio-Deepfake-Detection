import os
from huggingface_hub import snapshot_download

def download_esdd2():
    repo_id = "XuepingZhang/ESDD2-CompSpoof-V2"
    local_dir = "./data/CompSpoofV2"
    
    print(f"[*] Downloading {repo_id} to {local_dir}...")
    print("[*] This is a large dataset (283 hours). Please ensure you have sufficient space (~30-50GB).")
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir, 
            repo_type="dataset",
            # ignore_patterns=["*.parquet", "*.md", "*.txt"] # Only if you want raw audio files
        )
        print(f"[*] Done! Data is ready in {local_dir}")
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    download_esdd2()
