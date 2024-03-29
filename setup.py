import os
from codecs import open
from setuptools import find_packages, setup

exec(open('cvm/version.py').read())
setup(
    name='cvm',
    version=__version__,
    description='Computer Vision Models',
    url='https://github.com/ffiirree/cv-models',
    author='Liangqi Zhang',
    author_email='zhliangqi@gmail.com',
    python_requires='>=3.8',
    install_requires=[
        'torch >= 1.12',
        'torchvision',
        'fvcore',
        'torchinfo',
        'tqdm',
        'pycocotools',
        'nvidia-dali-cuda110 >= 1.16'
    ],
    packages=find_packages(exclude=['tests'])
)