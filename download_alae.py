import urllib.request, os, sys

def download(url, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r  {pct}% ({count*block_size//1024//1024} MB / {total_size//1024//1024} MB)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()
    print(f"Done: {dest}")

base = "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq"
root = os.path.dirname(os.path.abspath(__file__))
out  = os.path.join(root, "ALAE", "training_artifacts", "ffhq")

download(f"{base}/model_157.pth", os.path.join(out, "model_157.pth"))
