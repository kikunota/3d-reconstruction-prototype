#!/usr/bin/env python3
"""
Setup script for 3D Reconstruction project
"""

from setuptools import setup, find_packages

setup(
    name="3d-reconstruction",
    version="1.0.0",
    description="3D Reconstruction using Structure from Motion (SfM)",
    author="3D Vision Project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0", 
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "open3d>=0.17.0",
        "flask>=2.3.0",
        "werkzeug>=2.3.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "3d-reconstruct=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)