# setup.py - 
from setuptools import setup, find_packages

setup(
    name='vit_from_scratch',
    version='0.1.0',
    packages=find_packages(),  # This finds mha/
    install_requires=[
        'torch>=1.12',
    ],
)

