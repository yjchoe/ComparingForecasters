from setuptools import setup

setup(
    name='comparecast',
    version='0.0.0',
    packages=['comparecast', 'comparecast.data_utils'],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20',
        'scipy',
        'pandas>=1.0',
        'seaborn>=0.11',
        'tqdm',
        'jupyter',
        'openpyxl',
        'confseq>=0.0.6',
    ],
    url='https://github.com/yjchoe/ComparingForecasters',
    license='MIT',
    author='YJ Choe, Aaditya Ramdas',
    author_email='yjchoe@cmu.edu',
    description='Comparing Sequential Forecasters'
)
