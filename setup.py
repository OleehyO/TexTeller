from setuptools import setup, find_packages


# Define the base dependencies
install_requires = [
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "evaluate",
    "opencv-python",
    "ray[serve]",
    "accelerate",
    "tensorboardX",
    "nltk",
    "python-multipart",
    "augraphy",
    "streamlit==1.30",
    "streamlit-paste-button",
    "shapely",
    "pyclipper",

    "optimum[exporters]",
]

setup(
    name="texteller",
    version="0.1.2",
    author="OleehyO",
    author_email="1258009915@qq.com",
    description="A meta-package for installing dependencies",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/OleehyO/TexTeller",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
