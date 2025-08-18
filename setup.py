from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies are defined in install_requires below

setup(
    name="ddacs",
    version="1.0.0",
    author="Sebastian Baum",
    author_email="Sebastian.Baum@ias.uni-stuttgart.de",
    description="Deep Drawing and Cutting Simulations (DDACS) Dataset - Python interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "h5py>=3.8.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "python-dateutil>=2.8.0",
        "pytz>=2021.1",
        "six>=1.16.0",
        "darus @ git+https://github.com/BaumSebastian/DaRUS-Dataset-Interaction.git@v1.0.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.0.0",
        ],
        "examples": [
            "torch>=2.0.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "torch>=2.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "ruff>=0.0.250",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "full": [
            "torch>=2.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "plotly>=5.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "ruff>=0.0.250",
        ],
    },
    keywords="machine-learning, dataset, simulation, fem, sheet-metal-forming, deep-drawing, hdf5, materials-science",
    project_urls={
        "Bug Reports": "https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset/issues",
        "Source": "https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset",
        "Dataset": "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801",
        "Paper": "https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html",
    },
)