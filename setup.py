from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="impedance-agent",
    version="0.1.0",
    author="Richard Chukwu",
    author_email="",
    description="AI-powered CLI tool for electrochemical impedance spectroscopy analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richinex/impedance-agent",
    packages=find_packages(include=["impedance_agent", "impedance_agent.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.26.4",
        "scipy>=1.10.0",
        "matplotlib>=3.5.0",
        "pandas>=2.0.0",
        "jax[cpu]>=0.4.13",
        "jaxopt>=0.7",
        "pydantic>=2.4.0",
        "impedance>=1.7.0",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "typer>=0.9.0",
        "rich>=13.6.0",
        "aiofiles>=23.2.1",
        "asyncio>=3.4.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "tox>=4.11.0",
            "openpyxl>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "impedance-agent=impedance_agent.cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Typing :: Typed",
    ],
    include_package_data=True,
    package_data={
        "impedance_agent": [
            "py.typed",
            "config/*.yaml",
            "data/*.csv",
        ],
    },
    keywords=[
        "electrochemistry",
        "impedance",
        "spectroscopy",
        "EIS",
        "AI",
        "CLI",
    ],
    project_urls={
        "Bug Reports": "https://github.com/richinex/impedance-agent/issues",
        "Source": "https://github.com/richinex/impedance-agent",
    },
)
