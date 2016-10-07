# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

setup(
    name='saliencyfilters',
    version='0.0.1',
    description='A Python implementation of Saliency Filters method',
    long_description=open('README.rst', 'r').read(),
    author='Arif Qodari',
    author_email='arif.qodari@gmail.com',
    url='https://github.com/arifqodari/saliencyfilters',
    license=open('LICENSE', 'r').read(),
    packages=find_packages()
)
