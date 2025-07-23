from setuptools import setup, Extension
import numpy as np

ext_modules = [
    Extension(
        "lbfgs._lowlevel",
        sources=[
            "pylbfgs.c",
        ],
        include_dirs=[np.get_include()],
        # other flags as needed
    )
]

setup(
    name="PyLBFGS",
    version="0.2.0.16",
    ext_modules=ext_modules,
    install_requires=["numpy>=1.13.1", "cython>=3.1.2"],
)