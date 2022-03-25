
from setuptools import setup, find_packages
import os
# To use a consistent encoding
from codecs import open

import tp3hitparser

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
# with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()


setup(
    name="tp3hitparser",
    version=tp3hitparser.__version__,
    packages=find_packages(),
    install_requires=[
        "h5py>=2.8.0,<3.2.0",
        "matplotlib>=3.0.0,<4.0.0",
        "Pillow>9.0.0,<10.0.0",
        "tensorflow<2.7.0",
        "scipy>1.4.0,<2.0.0",
        "numpy>=1.16.0,<1.20.0",
        "tqdm>=4.0.0,<5.0",
        "mrcfile>1.0.0,<2.0.0",
    ],
    package_data={'tp3hitparser': [
        'default.cfg',
        '300.cfg']
        },
    author="Paul Van Schayck",
    # description="Add entries to start Jupyter from context menu.",
    # long_description=long_description,
    # long_description_content_type="text/x-rst",
    license="BSDv3",
    url="https://github.com/hyperspy/start_jupyter_cm",
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
            'tp3hitparser = tp3hitparser.tpx3HitParser:main',
        ], }
)