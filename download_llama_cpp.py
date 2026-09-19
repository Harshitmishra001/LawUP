import urllib.request
import json
import zipfile
import os

print("Fetching latest release...")
url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    
asset_url = None
for asset in data['assets']:
    if "bin-win-cpu-x64.zip" in asset['name']:
        asset_url = asset['browser_download_url']
        break

if not asset_url:
    print("Could not find asset")
    exit(1)

print(f"Downloading {asset_url}...")
urllib.request.urlretrieve(asset_url, "llama_cpp.zip")

print("Extracting...")
with zipfile.ZipFile("llama_cpp.zip", 'r') as zip_ref:
    zip_ref.extractall("llama_cpp_bin")
    
print("Done!")
