import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fullwaveqc",
    version="1.0.0",
    author="Deborah Pelacani Cruz",
    author_email="deborah.pelacani@gmail.com",
    description="A package for quality checking the inputs and outputs of Fullwave3D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="Nonefull",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)