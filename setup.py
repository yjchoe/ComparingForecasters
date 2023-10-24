from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='comparecast',
    version='1.0.0',
    packages=['comparecast', 'comparecast.data_utils'],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20',
        'scipy',
        'pandas>=1.0',
        'seaborn>=0.11',
        'tqdm',
        'openpyxl',
        'confseq>=0.0.9',
    ],
    url='https://github.com/yjchoe/ComparingForecasters',
    license='MIT',
    author='Yo Joong Choe, Aaditya Ramdas',
    author_email='yjchoe@uchicago.edu',
    description='Comparing Sequential Forecasters',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
