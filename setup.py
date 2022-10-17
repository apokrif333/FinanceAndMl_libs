from setuptools import find_packages, setup
from APIs.API_Tiingo import multithread

setup(
    name='FinanceAndMl_libs',
    packages=find_packages(include=['FinanceAndMl_libs']),
    version='0.1.0',
    description='Libs for Machine Learning and Finance Data Science',
    author='Me',
    license='MIT',
)
