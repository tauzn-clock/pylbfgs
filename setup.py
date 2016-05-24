from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension(
    	'pylbfgs',
    	sources=['pylbfgs.c'],
    	libraries=['lbfgs'],
    	library_dirs=['/usr/local/lib'],
    	include_dirs=['/usr/local/include'].extend(
    		numpy.distutils.misc_util.get_numpy_include_dirs()
    		),
    	runtime_library_dirs=['/usr/local/lib'],
    	)],
)
