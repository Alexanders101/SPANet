from setuptools import find_packages, setup

setup(
    name="spanet",
    packages=find_packages(),
    version="2.0.0",
    description="Symmetry Preserving Attention Networks",
    author="Alexander Shmakov",
    author_email="Alexanders101@gmail.com",
    url="https://github.com/Alexanders101/SPANet",
    license="BSD-3-Clause",
    install_requires=[
        'torch>=1.10',
        'numpy>=1.22',
        'h5py>=3.7',
        'numba>=0.55',
        'pytorch-lightning',
        'tqdm',
        'sympy',
        'pyyaml',
        'opt_einsum',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA :: 11.3',
        'Environment :: GPU :: NVIDIA CUDA :: 11.6',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Typing :: Typed'
    ],
)
