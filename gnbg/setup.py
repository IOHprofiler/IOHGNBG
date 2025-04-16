import os
import platform
from glob import glob
from setuptools import setup, find_packages

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.1.3"

ext = Pybind11Extension(
    "gnbg.gnbgcpp", ["interface.cpp"], include_dirs=["."], cxx_std=17,
)

if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    ext._add_cflags(["-O3"])
else:
    ext._add_cflags(["/O2"])


setup(
    name="gnbg",
    author="Jacob de Nobel",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    long_description_content_type="text/markdown",
    packages=find_packages(),
    zip_safe=False,
    version=__version__,
    install_requires=[
        "ioh", 
        "numpy", 
    ],
    include_package_data=True,
)
