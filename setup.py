import os
import sys
import numpy

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

if sys.platform == 'linux':
    print('compiling for linux using gcc')
    compile_args = ['-fopenmp', '-std=c17', '-march=native']
    link_args=['-fopenmp']
elif sys.platform == 'win32':
    print('compiling for windows')
    compile_args = ['-openmp']
    link_args = []
else:
    print(f'Do not recognise system platform: {sys.platform}')



os.chdir(os.path.dirname(os.path.abspath(__file__)))

extensions = [
    Extension(
        'slidepy._ext',
        [
            'slidepy/_ext.pyx',
            'slidepy/augment/augment.c',
            'slidepy/com/com.c',
            'slidepy/mp_math/mp_math.c',
            ],
        include_dirs=[numpy.get_include()],
        extra_compile_args = compile_args,
        extra_link_args = link_args
        #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        #language="c++"
        )
    ]


# Make sure everything is compiled with pyton 3
for e in extensions:
    e.cython_directives = {'language_level': '3'}

# load readme
with open('README.md', mode = 'rb') as readme_file:
    readme = readme_file.read().decode('utf-8')

setup(
    name = 'slidepy',
    version='0.0.1',
    description='Fast multi-threaded 3D landslide modelling with SIMD support',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/asenogles/slidepy',
    author='asenogles',
    license='LGPL v3',
    packages=find_packages(),
    install_requires=[
        'cython>=0.29.21',
        'numpy>=1.19.0',
        'scipy>=1.7.0'
        ],
    python_requires='>=3.6',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
)
