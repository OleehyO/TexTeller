from setuptools import setup, find_packages
import platform

# Define the base dependencies
install_requires = [
    "torch",
    "torchvision",
    "torchaudio",
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
]

# Add platform-specific dependencies
if platform.system() == "Darwin":
    install_requires.append("onnxruntime")
else:
    install_requires.append("onnxruntime-gpu")

setup(
    name="texteller",
    version="0.1.0",
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
