from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.0.4',
                     'h5py==2.7.0',
                     'sklearn']

setup(
    name='iris',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
