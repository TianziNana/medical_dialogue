from setuptools import setup, find_packages

setup(
    name="fairness-aware-medical-summarization",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.21.0",
        "datasets>=2.0.0",
        "peft>=0.4.0",
        "scikit-learn>=1.1.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "spacy>=3.4.0",
        "rouge-score>=0.1.2",
        "tqdm>=4.64.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Fairness-aware medical text summarization with counterfactual data augmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fairness-aware-medical-summarization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
