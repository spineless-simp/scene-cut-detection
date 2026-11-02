"""Setup script for scene-cut-detector."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the version from the package
version_path = Path(__file__).parent / "scene_cut_detector" / "__init__.py"
version = None
with open(version_path) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
            break

if not version:
    raise ValueError("Could not find version in __init__.py")

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
with open(readme_path) as f:
    long_description = f.read()

setup(
    name="scene-cut-detector",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Detect hard scene cuts in video files using FFmpeg streaming",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scene-cut-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "scene-cut-detector=scene_cut_detector.cli:main",
        ],
    },
)
