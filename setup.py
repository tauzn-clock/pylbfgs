from distutils.core import setup, Extension
import numpy.distutils.misc_util
from version import version

ext_modules = [
    Extension(
        'pylbfgs',
        sources=['pylbfgs.c'],
        libraries=['lbfgs'],
        library_dirs=['/usr/local/lib'],
        include_dirs=['/usr/local/include'].extend(
            numpy.distutils.misc_util.get_numpy_include_dirs()
            ),
        runtime_library_dirs=['/usr/local/lib'],
        ),
    ]

setup(
    name='PyLBFGS',
    version=version,
    author='Robert Taylor',
    author_email='rtaylor@pyrunner.com',
    url='https://bitbucket.org/rtaylor/pylbfgs',
    description='PyLBFGS is a Python 3 wrapper of the libLBFGS library written by Naoaki Okazaki.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        ],
    ext_modules=ext_modules,
)
