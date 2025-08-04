from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="smart-product-classifier",
    version="1.0.0",
    author="Stanley Ndaba",
    author_email="your.email@example.com",
    description="A production-ready Computer Vision API for real-time e-commerce product classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanleyndaba/product-classifier-api",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-classifier-api=src.services.api:main",
            "smart-classifier-train=src.training.train:main",
            "smart-classifier-predict=src.inference.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.conf"],
    },
)