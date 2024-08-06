from setuptools import setup, find_packages

setup(
    name="rag-evaluator-metrics",
    version="0.1.0",
    description='''This library provides a comprehensive suite of metrics to evaluate the performance of Retrieval-Augmented Generation (RAG) systems. RAG systems, which combine information retrieval with text generation, present unique evaluation challenges beyond those found in standard language generation tasks''',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Beekash Mohanty",
    author_email="beekashmohanty222@gmail.com",
    url="https://github.com/AIAnytime/rag-evaluator",
    packages=find_packages(),
    install_requires=[
        "torch",
        "sacrebleu",
        "rouge-score",
        "bert-score",
        "transformers",
        "nltk",
        "textblob",
        "textstat" 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)