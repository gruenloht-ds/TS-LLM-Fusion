import os

from tqdm import tqdm
import wget
from ts_llm_fusion.core.configs import load_configs

configs = load_configs()

with open(configs.paths.download_datasets, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

for url in urls:
    wget.download(url, out=configs.paths.real)
    print()
