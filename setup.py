"""
Cultural Impact Assessment Tool (CIAT) - Package Setup

This script handles the installation of the CIAT package, including dependencies
and entry point scripts.

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025

References:
    - https://flask-script.readthedocs.io/en/latest/
    - https://packaging.python.org/en/latest/tutorials/packaging-projects/
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://github.com/mbr/pypi-classifiers/blob/master/setup.py
    - https://github.com/levan92/logging-example
    - https://github.com/DataCanvasIO/DeepTables/blob/master/setup.py
"""

# Standard library imports
import os
from setuptools import setup, find_packages


def read_requirements():
    """
    Read the requirements file and return a list of dependencies.
    
    This function reads the requirements.txt file, filters out comments
    and empty lines, and returns a list of package dependencies.
    
    Returns:
        list: List of package dependencies
    """
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path) as f:
            # Filter out comments and empty lines
            return [line for line in f.read().splitlines() if line and not line.startswith('#')]
    except FileNotFoundError:
        print("Warning: requirements.txt not found")
        return []


setup(
    name="cultural-impact-tool",
    version="1.0.0",
    description="A predictive model for assessing cultural impact on project success",
    long_description="""
    The Cultural Impact Assessment Tool (CIAT) is a predictive model and assessment
    tool for determining the extent of cultural impact on international project
    management success. Based on the research framework developed by Hainadine Chamane,
    it draws on cultural variables identified in Fog's (2022) cross-cultural study and
    Dumitrașcu-Băldău, Dumitrașcu and Dobrotă's (2021) research on factors influencing
    international project success.
    """,
    author="Hainadine Chamane",
    author_email="hchamane@outlook.com",
    url="https://github.com/hchamane/CIAT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'ciat=run:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Framework :: Flask',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
)
