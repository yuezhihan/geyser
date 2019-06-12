import os
import sys
from setuptools import setup
from .geyser.VERSION import VERSION

def _read(file):
    with open(file, 'rb') as fp:
        return fp.read()

if sys.argv[-1] == 'publish':
    os.system("rm -r dist/")
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    sys.exit()

# TODO: rename to firebolt
setup(name='geyser',
      version=VERSION,
      description='geyser',
      long_description=_read('README.md').decode('utf-8'),
      long_description_content_type='text/markdown',
      author='Zhihan Yue',
      author_email='zhihan.yue@foxmail.com',
      url='https://github.com/yuezhihan/geyser',
      python_requires=">=2.7",
      install_requires=['torch'],
      packages=['geyser'],
      classifiers=[
          'License :: OSI Approved :: MIT License'
      ],
      zip_safe=False)
