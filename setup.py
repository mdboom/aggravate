# -*- coding: utf-8 -*-

# Copyright (c) 2015, Michael Droettboom All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and
# documentation are those of the authors and should not be interpreted
# as representing official policies, either expressed or implied, of
# the FreeBSD Project.

from __future__ import print_function, absolute_import

import versioneer

from distutils.core import setup, Extension

import glob
import os
import sys

PY3 = (sys.version_info[0] >= 3)

sys.path.insert(0, 'buildext')
sys.path.insert(0, 'doc')

import convert_docstrings

import numpy as np

if __name__ == '__main__':
    convert_docstrings.convert_all_docstrings('docstrings', 'src/doc')

    cmdclass = versioneer.get_cmdclass()

    agg_sources = ['agg_image_filters.cpp', 'agg_trans_affine.cpp']

    agg_sources = [
        os.path.join('extern', 'agg24-svn', 'src', x) for x in agg_sources]

    extra_compile_args = []
    libraries = []
    define_macros = [('PY_ARRAY_UNIQUE_SYMBOL', 'AGGRAVATE_ARRAY_API')]

    if sys.platform == 'linux2':
        extra_compile_args.append('-fopenmp')
        libraries.append('libgomp')
        define_macros.append(('USE_OPENMP', 1))

    extension = Extension(
        'aggravate._aggravate',
        glob.glob('src/*.cpp') +
        glob.glob('src/doc/*.c') +
        agg_sources,
        define_macros=define_macros,
        include_dirs=[np.get_include(), 'extern/agg24-svn/include'],
        extra_compile_args=extra_compile_args,
        libraries=libraries
        )

    setup(name="aggravate",
          version=versioneer.get_version(),
          description="2D image interpolation library",
          author="Michael Droettboom",
          author_email="mike@droettboom.com",
          packages=['aggravate', 'aggravate.tests'],
          package_dir={'': 'lib'},
          ext_modules=[extension],
          cmdclass=cmdclass)
