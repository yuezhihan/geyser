from setuptools import setup
def _read(file):
    with open(file, 'rb') as fp:
        return fp.read()

setup(name='geyser',
      version='0.1',
      description='geyser',
      long_description=_read('README.md').decode('utf-8'),
      author='Zhihan Yue',
      author_email='zhihan.yue@foxmail.com',
      url='https://github.com/yuezhihan/geyser',
      python_requires=">=2.7",
      install_requires=["torch"],
      packages=["geyser"],
      classifiers=[
          'License :: OSI Approved :: MIT License'
      ]
      zip_safe=False)
