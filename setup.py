import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smaPy",
    version="0.0.1",
    author="Huan Chen",
    author_email="hchen130@jhu.edu",
    packages=["smaPy"],
    description="A Python package implements structrally masked autoencoder, a deep autoencoder for joint decomposing of multiple data sets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CHuanSite/smaPy",
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
         "numpy",
         "pandas",
         "torch"
    ]
)
