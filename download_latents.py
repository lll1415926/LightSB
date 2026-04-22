import gdown, os

root = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(root, "data")
os.makedirs(data, exist_ok=True)

files = {
    "latents.npy":     "https://drive.google.com/uc?id=1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
    "gender.npy":      "https://drive.google.com/uc?id=1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf",
    "age.npy":         "https://drive.google.com/uc?id=1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
    "test_images.npy": "https://drive.google.com/uc?id=1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i",
}

for name, url in files.items():
    dest = os.path.join(data, name)
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        continue
    print(f"\n=== Downloading {name} ===")
    gdown.download(url, dest, quiet=False)

print("\nAll done!")
