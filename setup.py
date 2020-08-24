from setuptools import setup, find_packages
from pathlib import Path


NAME = "feature_raster"
DESCRIPTION = "Feature engineering for satellite image"
EMAIL = "hectorpatino24@gmail.com"
AUTHOR = "Hector Patiño"
REQUIRES_PYTHON = ">=3.8.3"
URL = "https://github.com/hectorpatino/feature_raster.git"

with open("README.md", "r") as fh:
    long_description = fh.read()


def list_req(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()


about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "feature_raster"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


setup(
    name=NAME,
    version=about["__version__"],
    description=long_description,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests", )),
    package_data={"feature_raster": ["VERSION"]},
    license="BSD 3 clause",
    install_requires=list_req(),
    include_package_data=True,
    classifiers=[
          # Trove classifiers
          # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
      ],
    zip_safe=False
)