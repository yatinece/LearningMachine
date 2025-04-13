from setuptools import setup, find_packages

setup(
    name="learnai_ready",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "pyyaml",
        "typer",
        "python-multipart",
        "streamlit"
    ],
    python_requires=">=3.8",
)
