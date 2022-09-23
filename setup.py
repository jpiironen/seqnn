from setuptools import setup, find_packages


with open('requirements.txt', 'r') as file:
    requirements = file.read().splitlines()

#version = (
#    open("lab/__init__.py").readlines()[-1].split("=")[-1].strip().strip("'\"")
#)

setup(
    name="seqnn",
    description="",
    #version=version,
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)
