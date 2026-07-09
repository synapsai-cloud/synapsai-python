from setuptools import setup, find_packages

setup(
    name="synapsai-python",
    version="0.1.0",
    description="The official SynapsAI Cloud Python SDK",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/synapsai-cloud/synapsai",
    author="SynapsAI Technologies Inc.",
    author_email="support@synapsai.cloud",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.23.0",
        "pydantic>=2.0,<3",
        "typing-extensions>=4.5",
        "Pillow>=12.2.0",
    ],
)
