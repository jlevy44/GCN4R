from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
PACKAGES=[]

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()


setup(name='gcn4r',
      version='0.1.1',
      description='Code to accompany GCN4R package.',
      url='https://github.com/jlevy44/GCN4R',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':[]
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['gcn4r'],
      install_requires=PACKAGES)
