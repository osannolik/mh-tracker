import os.path
from setuptools import (find_packages, setup)

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name="mht",
    version="0.0.1",
    description="Track-oriented multiple hypothesis tracker",
    keywords=['MOT', 'MTT', 'MHT', 'JPDA', 'PMBM', 'tracking'],
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/osannolik/mh-tracker",
    author="osannolik",
    author_email="osannolik.pypi@godtycklig.se",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    install_requires=[
        "numpy", "matplotlib", "motmetrics"
    ],
)