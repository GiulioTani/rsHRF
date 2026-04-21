import re
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# Using .strip() ensures no hidden newline or carriage return characters (\r) 
# remain in the version string, which is a common cause for 400 errors.
with open("rsHRF/VERSION", "r") as fh:
    VERSION = fh.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version when releasing to a tag."""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag is not None and tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)    
    
setup(
    name="rsHRF",
    packages=find_packages(),
    entry_points={
        "console_scripts": ['rsHRF = rsHRF.CLI:main']
    },
    version=VERSION,
    description="BIDs App to retrieve the haemodynamic response function from resting state fMRI data",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Madhur Tandon, Amogh Johri",
    # Using a single primary email often helps avoid validation parsing issues on PyPI
    author_email="madhurtandon23@gmail.com",
    url="https://github.com/BIDS-Apps/rsHRF",
    # Changed from tuple () to list [] to comply with standard setuptools expectations
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy", 
        "nibabel", 
        "matplotlib", 
        "scipy", 
        "pybids==0.11.1", 
        "pandas", 
        "patsy", 
        "mpld3", 
        "duecredit", 
        "joblib", 
        "PyWavelets"
    ],
    cmdclass={
        'verify': VerifyVersionCommand,
    },
)