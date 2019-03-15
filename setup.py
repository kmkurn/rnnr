from pathlib import Path
from setuptools import setup, find_packages
import re

here = Path(__file__).resolve().parent
readme = (here / 'README.rst').read_text()
version = re.search(
    r'__version__ = (["\'])([^"\']*)\1',
    (here / 'rnnr' / '__init__.py').read_text(),
)[2]

setup(
    name='rnnr',
    version=version,
    description='Runner for neural network training or evaluation',
    long_description=readme,
    url='https://github.com/kmkurn/rnnr',
    author='Kemal Kurniawan',
    author_email='kemal@kkurniawan.com',
    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    python_requires='>=3.6, <4',
)
