from setuptools import setup, find_packages
setup(name='lenet5mnist',
      scripts=['lenet5mnist/train-lenet5-mnist'],
      package_dir={'lenet5mnist': 'lenet5mnist'},
      packages=find_packages())
