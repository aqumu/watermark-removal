import argparse
from watermark_gen.core.config import load_config
from watermark_gen.core.downloader import download_images
from watermark_gen.core.dataset import generate_dataset
from watermark_gen.utils.random import seed_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_all(config["seed"])

    download_images(config)

    generate_dataset(config)