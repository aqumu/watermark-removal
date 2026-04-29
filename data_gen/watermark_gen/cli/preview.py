import cv2
from pathlib import Path
import random

def main():
    dataset = Path("./dataset")
    samples = list(dataset.glob("sample_*"))
    sample = random.choice(samples)

    img = cv2.imread(str(sample / "watermarked.jpg"))
    cv2.imshow("Preview", img)
    cv2.waitKey(0)