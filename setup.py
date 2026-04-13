#!/usr/bin/env python
"""
Setup script for fire detection system
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="fire-detection-system",
    version="1.0.0",
    description="Real-time fire detection using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fire Detection Team",
    author_email="info@firedetection.local",
    url="https://github.com/yourname/fire-detection-system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "imutils>=0.5.4",
    ],
    entry_points={
        "console_scripts": [
            "fire-detection=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
