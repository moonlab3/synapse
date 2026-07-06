from setuptools import setup, find_packages

setup(
    name='synapse',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Young',
    maintainer_email='yh_moon@wonik.com',
    description='Synapse. ROS2 Wrapper',
    license='License',
    tests_require=['pytest'],
)