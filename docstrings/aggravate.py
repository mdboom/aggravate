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

from __future__ import absolute_import

resample = """
resample(input_array, output_array, sx, shy, shx, sy, tx=0, ty=0, method=NEAREST, norm=1, radius=1)

Resample input_array, blending it in-place into output_array, using an
affine transformation and a given interpolation method.

Parameters
----------
input_array : 2-d Numpy array of double

output_array : 2-d Numpy array of double
    The resampled input array is blended into the output array.  It
    should be initialized with zeros.

sx, shy, shx, sy, tx, ty : float
    The affine transformation from the input array to the output
    array.

    *sx* and *sy* are scale, *shx* and *shy* are shear, and *tx* and
    *ty* are translation.

method : int, optional
    The interpolation method.  Must be one of the following constants
    defined in this module:

      NEAREST (default), BILINEAR, BICUBIC, SPLINE16, SPLINE36,
      HANNING, HAMMING, HERMITE, KAISER, QUADRIC, CATROM, GAUSSIAN,
      BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN

norm : float, optional
    The norm for the interpolation function.  Default is 0.

radius: float, optional
    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.
    Default is 1.
"""
