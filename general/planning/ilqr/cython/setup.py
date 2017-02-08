# setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(ext_modules=[Extension("ilqr_backward",
                             ["ilqr_backward.pyx", "c_ilqr_backward.cpp"],
                             language="c++",
                             include_dirs=['/usr/include', '/usr/include/eigen3'])],
      cmdclass = {'build_ext': build_ext})

# setup(ext_modules = cythonize(
#            "cost_torque.pyx",                 # our Cython source
#            sources=["cost_torque_cpp.cpp"],  # additional source file(s)
#            language="c++",             # generate C++ code
#       ))
