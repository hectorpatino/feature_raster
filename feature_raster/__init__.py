import pathlib
import feature_raster

PACKAGE_ROOT = pathlib.Path(feature_raster.__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / 'VERSION'

name = "feature_raster"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()