from setuptools import setup, find_packages

setup(
    name="synthetic_watermark_generator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "pyyaml",
        "tqdm",
        "scipy",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "wm-generate=watermark_gen.cli.generate:main",
            "wm-preview=watermark_gen.cli.preview:main",
        ]
    },
)