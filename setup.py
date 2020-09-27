#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date    : 2020/6/29
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com

from setuptools import setup, find_packages

setup(
    name='toolkit4nlp',
    version='0.3.0',
    description='an toolkit for nlp research',
    long_description='toolkit4nlp: https://github.com/xv44586/toolkit4nlp',
    license='Apache License 2.0',
    url='https://github.com/xv44586/toolkit4nlp',
    author='xv44586',
    author_email='xv44586@gmail.com',
    install_requires=['keras<=2.3.1'],
    packages=find_packages()
)
