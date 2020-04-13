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
    author_email="osannolik@godtycklig.se",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=("test",)),
    include_package_data=False,
    install_requires=[
        "numpy", "matplotlib", "motmetrics"
    ],
)



from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='mht',
    version='0.0.1',
    description='Track-Oriented Multiple Hypothesis Tracker',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='osannolik',
    author_email='osannolik@godtycklig.se',
    keywords=['MHT', 'MOT', 'MTT', 'Tracker'],
    url='https://github.com/osannolik/mh-tracker/',
    download_url='https://pypi.org/project/mh-tracker/'
)

install_requires = [
    'elasticsearch>=6.0.0,<7.0.0',
    'jinja2'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)