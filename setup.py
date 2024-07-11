import os
from setuptools import setup, find_packages

version = '0.0.2'

with open('README.md') as f:
    long_description = f.read()

setup(
    name='flashdp',
    version=version,
    description='Packages of Flash Differential Privacy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/FlashDPSGD/FlashDP4review/",
    author='FlashDPSGD',
    packages=find_packages(),
    python_requires=">=3.11.0",
    include_package_data=True,
    extras_require={
        "test": [
            "tqdm>=4.62.3",
            "transformers>=4.38",
            "opacus>=1.4",
        ]
    },
    install_requires=[
        "torch>=2.2.0",
        "triton>=2.2.0",
    ],
    test_suite="tests",
    zip_safe=False
)
