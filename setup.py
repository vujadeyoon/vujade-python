"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: setyo.py
Description: A python3 script for setup the python3 package, vujade.

Reference:
    i)  https://github.com/pytorch/vision/blob/master/setup.py
    ii) https://github.com/1adrianb/face-alignment/blob/master/setup.py
"""


import io
import os
import re
from setuptools import find_packages, setup


class SetupTools(object):
    @staticmethod
    def _read(*names, **kwargs) -> str:
        with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get('encoding', 'utf8')) as fp:
            res = fp.read()
        return res

    @classmethod
    def get_python3_package_version(cls, *file_paths) -> str:
        version_file = cls._read(*file_paths)
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
        if version_match:
            res = version_match.group(1)
        else:
            raise ValueError('The required python3 package version cannot be detected.')
        return res


setup(
    name='vujade',
    version=SetupTools.get_python3_package_version('vujade', '__init__.py'),
    descrption='A collection of useful Python3 classes and functions for deep learning research and development.',
    author='vujadeyoon',
    author_email='vujadeyoon@gmail.com',
    url='https://github.com/vujadeyoon/vujade-python',
    license='MIT License',
    packages=find_packages(exclude=[]),
    install_requires=['Cython', 'numpy', 'wheel'],
    keywords=['vujade-python', 'vujade'],
    python_requires='>=3.8',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
