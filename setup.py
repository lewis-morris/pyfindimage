from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='pyfindimage',
    version='0.3.05',
    packages=['locater'],
    url='',
    license='MIT',
    author='lewis',
    author_email='lewis.morris@gmail.com',
    description='Image finder based on text',
    install_requires=requirements,
)
