import os
from huggingface_hub import snapshot_download

def download_esdd2():
    repo_id   = "XuepingZhang/ESDD2-CompSpoof-V2"
    local_dir = "./data/CompSpoofV2"
    print(f"[*] Downloading {repo_id} to {local_dir} ...")
    print("[*] Large dataset (~283 hours). Ensure ~50 GB free space.")
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset")
        print(f"[*] Done! Data ready in {local_dir}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_esdd2()
