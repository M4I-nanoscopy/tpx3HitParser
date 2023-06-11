
from setuptools import setup, find_packages
import os
# To use a consistent encoding
from codecs import open

import tpx3hitparser

here = os.path.abspath(os.path.dirname(__file__))


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tpx3hitparser",
    version=tpx3hitparser.__version__,
    packages=find_packages(),
    install_requires=[
        "h5py>=3.0.0,<4.0.0",
        "matplotlib>=3.0.0,<4.0.0",
        "Pillow>9.0.0,<10.0.0",
        "tensorflow<2.9.0",
        "scipy>1.4.0,<2.0.0",
        "numpy>=1.20.0,<1.25.0",
        "tqdm>=4.0.0,<5.0",
        "mrcfile>1.0.0,<2.0.0",
    ],
    package_data={'tpx3hitparser': [
        'default.cfg',
        '300.cfg',
        '200kv-events-chip_edge.cfg']
        },
    author="Paul Van Schayck",
    description="Convert TIMEPIX3 files to hdf5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/M4I-nanoscopy/tpx3HitParser",
    project_urls={
        "Bug Tracker": "https://github.com/M4I-nanoscopy/tpx3HitParser/issues",
    },
    license="BSDv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Desktop Environment :: Gnome",
    ],
    entry_points={
        'console_scripts': [
            'tpx3hitparser = tpx3hitparser.tpx3HitParser:main',
        ], }
)