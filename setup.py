from setuptools import setup, find_packages

requirements = [
    "tqdm",
    "kaldiio",
    "torch>=1.12.0",
    "torchaudio>=0.12.0",
    "silero-vad",
]

setup(
    name="wesep",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wesep = wesep.cli.extractor:main",
        ],
    },
)
