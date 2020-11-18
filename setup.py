import setuptools
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

requirements = [
    "pandas >= 1.0.1",
    "numpy >= 1.0.0",
    "scipy >= 1.1.0",
    "matplotlib >= 1.5",
    "seaborn >= 0.11.0",
]

setuptools.setup(name='hypnospy',
        version='0.0.3',
        author='Ignacio Pozuelo, Joao Palotti',
        author_email='ignacio_perez_pozuelo@alumni.brown.edu, joaopalotti@gmail.com',
        license='BSD',
        install_requires=requirements,
        packages=setuptools.find_packages(),
        #package_dir = {'': '.'},
        url='https://github.com/ippozuelo/HypnosPy/',
        description='A Device-Agnostic, Open-Source Python Software for Wearable Circadian Rhythm and Sleep Analysis and Visualization',
        long_description_content_type="text/markdown",
        long_description=open('README.md').read(),
        python_requires='>=3.6',
)

