from distutils.core import setup

from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

requirements = [
    "pandas >= 1.0.1",
    "numpy >= 1.0.0",
    "scipy >= 1.1.0",
    #"matplotlib >= 1.5",
    #"seaborn >= 0.10.1",
]

setup(name='hypnospy',
        version='0.0.1',
        author='Ignacio Pozuelo, Joao Palotti',
        author_email='ignacio_perez_pozuelo@alumni.brown.edu, joaopalotti@gmail.com',
        license='BSD',
        install_requires=requirements,
        packages=['hypnospy'],
        #package_dir = {'': '.'},
        url='https://github.com/joaopalotti/hypnospy',
        description='A Device-Agnostic, Open-Source Python Software for Wearable Circadian Rhythm and Sleep Analysis and Visualization',
        long_description=open('README.md').read()
)

