from setuptools import setup

package_name = 'imitation_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='justin kim',
    maintainer_email='justinkim0345@gmail.com',
    description='f1tenth imitation_pkg',
    license='UBC',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imitation_node = imitation_pkg.imitation_node:main',
        ],
    },
)
