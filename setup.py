from setuptools import setup, find_packages


with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

with open("requirements_gym.txt", "r") as file:
    requires_gym = file.read().splitlines()

# version = (
#    open("seqnn/__init__.py").readlines()[-1].split("=")[-1].strip().strip("'\"")
# )

setup(
    name="seqnn",
    description="",
    # version=version,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"gym": requires_gym},
)
