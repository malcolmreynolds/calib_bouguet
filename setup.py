from setuptools import setup
import numpy as np

setup(
    name="calib_bouguet",
    version="0.0.1",
    description="Load the results of the Matlab Bouguet calibration toolbox",
    author="Malcolm Reynolds",
    author_email="malcolm.reynolds@gmail.com",
    packages=["calib_bouguet"],
    install_requires=['numpy', 'scipy'],
)
