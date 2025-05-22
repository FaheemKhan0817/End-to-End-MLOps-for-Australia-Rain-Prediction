from setuptools import setup, find_packages

# Read requirements from requirements.txt
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []  

setup(
    name="Australia Rain Prediction System",  
    version="0.1",
    author="Faheem Khan",
    author_email="faheemthakur23@gmail.com",
    description="End to End MLOps Project for Australia Rain Prediction System",
    packages=find_packages(),
    install_requires=requirements
)

