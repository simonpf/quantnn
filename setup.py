import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qrnn",
    version="0.0.1",
    author="Simon Pfreundschuh",
    description="Quantile regression neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonpf/qrnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU Affero",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy"],
    tests_require=["sphinx_rtd_theme", "pytest"],
    python_requires=">=3.6",
)
