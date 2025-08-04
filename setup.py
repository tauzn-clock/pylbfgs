from setuptools import setup, Extension
import numpy as np

ext_modules = [
    Extension(
        "pylbfgs",
        sources=["pylbfgs.c"],
        libraries=['lbfgs'],
        library_dirs=['/usr/local/lib'],
        include_dirs=['/usr/local/include'] + [np.get_include()],
        runtime_library_dirs=['/usr/local/lib'],
    )
]

setup(
    name='PyLBFGS',
    version="0.2.0.16",
    ext_modules=ext_modules,
    install_requires=["cython>=3.1.2"],
)