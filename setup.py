from setuptools import find_packages, setup 



setup(
    name="DiamondPricePrediction", 
    version="0.0.1", 
    author="Fiawornu Etor Peniel", 
    author_email="penieletornam18@gmail.com", 
    install_requires = ["scikit-learn", "pandas", "numpy"],
    packages=find_packages()
)