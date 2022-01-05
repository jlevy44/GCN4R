from setuptools import setup
from setuptools.command.install import install
import subprocess
import os

PACKAGES=['pysnooper',
            'fire',
            'numpy==1.18.1',
            'pandas==0.25.3',
            'networkx>=2.4',
            'torch==1.4.0',
            'scikit-learn==0.22.1',
            'scipy==1.4.1',
            'torch-scatter==2.0.4',
            'torch-cluster==1.5.4',
            'torch-sparse==0.6.1',
            'torch-geometric==1.5.0',
            'torch_spline_conv==1.2.0',
            'torchvision==0.5.0',
            'matplotlib==3.1.2',
            'seaborn==0.10.0',
            'plotly==4.5.0',
            'rpy2==2.9.4',
            'cdlib>=0.1.8',
            'captum==0.2.0']

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

setup(name='gcn4r',
      version='0.1',
      description='Code to accompany GCN4R package.',
      url='https://github.com/jlevy44/GCN4R',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=['bin/install_gcn4r_dependencies'],
      entry_points={
            'console_scripts':['gcn4r=gcn4r.cli:main']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['gcn4r'],
      install_requires=PACKAGES,
      package_data={'gcn4r': ['data/*']})
