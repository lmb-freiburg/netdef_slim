from setuptools import find_packages, setup

setup(
    name='netdef_slim',
    description='A python wrapper for tf to ease creation of network definitions',
    version='',
    url='https://github.com/lmb-freiburg/netdef_slim',
    license='GPLv3.0',
    author='lmb-freiburg',
    author_email='',
    packages=find_packages(),
    install_requires=[
        'lmbspecialops',
        'tensorflow',
        'scikit-learn',
        'pillow',
        'scipy'
    ],
)
