/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
Copyright (c) 2015, Michael Droettboom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

/* agg does not have built-in support for double-precision floating
   point images.  This is a modification of the existing gray32 (for
   single-precision) to become gray64 (for double-precision) */

#ifndef GRAY64_H
#define GRAY64_H

#include "agg_color_gray.h"

// Based on:

//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software
// is granted provided this copyright notice appears in all copies.
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Adaptation for high precision colors has been sponsored by
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
//

//===================================================================gray64
namespace agg
{
    struct gray64
    {
        typedef double value_type;
        typedef double calc_type;
        typedef double long_type;
        typedef gray64 self_type;

        value_type v;
        value_type a;

        // Calculate grayscale value as per ITU-R BT.709.
        static value_type luminance(double r, double g, double b)
        {
            return value_type(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }

        static value_type luminance(const rgba& c)
        {
            return luminance(c.r, c.g, c.b);
        }

        static value_type luminance(const rgba32& c)
        {
            return luminance(c.r, c.g, c.b);
        }

        static value_type luminance(const rgba8& c)
        {
            return luminance(c.r / 255.0, c.g / 255.0, c.g / 255.0);
        }

        static value_type luminance(const rgba16& c)
        {
            return luminance(c.r / 65535.0, c.g / 65535.0, c.g / 65535.0);
        }

        //--------------------------------------------------------------------
        gray64() {}

        //--------------------------------------------------------------------
        explicit gray64(value_type v_, value_type a_ = 1) :
        v(v_), a(a_) {}

        //--------------------------------------------------------------------
    gray64(const self_type& c, value_type a_) :
        v(c.v), a(a_) {}

        //--------------------------------------------------------------------
    gray64(const rgba& c) :
        v(luminance(c)),
            a(value_type(c.a)) {}

        //--------------------------------------------------------------------
    gray64(const rgba8& c) :
        v(luminance(c)),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
    gray64(const srgba8& c) :
        v(luminance(rgba32(c))),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
    gray64(const rgba16& c) :
        v(luminance(c)),
            a(value_type(c.a / 65535.0)) {}

        //--------------------------------------------------------------------
    gray64(const rgba32& c) :
        v(luminance(c)),
            a(value_type(c.a)) {}

        //--------------------------------------------------------------------
    gray64(const gray8& c) :
        v(value_type(c.v / 255.0)),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
    gray64(const gray16& c) :
        v(value_type(c.v / 65535.0)),
            a(value_type(c.a / 65535.0)) {}

        //--------------------------------------------------------------------
    gray64(const gray32& c) :
        v(value_type(c.v)),
            a(value_type(c.a)) {}

        //--------------------------------------------------------------------
    gray64(const gray64& c) :
        v(c.v),
            a(c.a) {}

        //--------------------------------------------------------------------
        operator rgba() const
        {
            return rgba(v, v, v, a);
        }

        //--------------------------------------------------------------------
        operator gray8() const
        {
            return gray8(uround(v * 255.0), uround(a * 255.0));
        }

        //--------------------------------------------------------------------
        operator gray16() const
        {
            return gray16(uround(v * 65535.0), uround(a * 65535.0));
        }

        //--------------------------------------------------------------------
        operator gray32() const
        {
            return gray32(v, a);
        }

        //--------------------------------------------------------------------
        operator rgba8() const
        {
            rgba8::value_type y = uround(v * 255.0);
            return rgba8(y, y, y, uround(a * 255.0));
        }

        //--------------------------------------------------------------------
        operator rgba16() const
        {
            rgba16::value_type y = uround(v * 65535.0);
            return rgba16(y, y, y, uround(a * 65535.0));
        }

        //--------------------------------------------------------------------
        operator rgba32() const
        {
            return rgba32(v, v, v, a);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE double to_double(value_type a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type from_double(double a)
        {
            return value_type(a);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type empty_value()
        {
            return 0;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type full_value()
        {
            return 1;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_transparent() const
        {
            return a <= 0;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_opaque() const
        {
            return a >= 1;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type invert(value_type x)
        {
            return 1 - x;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type multiply(value_type a, value_type b)
        {
            return value_type(a * b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type demultiply(value_type a, value_type b)
        {
            return (b == 0) ? 0 : value_type(a / b);
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downscale(T a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downshift(T a, unsigned n)
        {
            return n > 0 ? a / (1 << n) : a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type mult_cover(value_type a, cover_type b)
        {
            return value_type(a * b / cover_mask);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return cover_type(uround(a * b));
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a, assuming q is premultiplied by a.
        static AGG_INLINE value_type prelerp(value_type p, value_type q, value_type a)
        {
            return (1 - a) * p + q; // more accurate than "p + q - p * a"
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a.
        static AGG_INLINE value_type lerp(value_type p, value_type q, value_type a)
        {
            // The form "p + a * (q - p)" avoids a multiplication, but may produce an
            // inaccurate result. For example, "p + (q - p)" may not be exactly equal
            // to q. Therefore, stick to the basic expression, which at least produces
            // the correct result at either extreme.
            return (1 - a) * p + a * q;
        }

        //--------------------------------------------------------------------
        self_type& clear()
        {
            v = a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            else if (a_ > 1) a = 1;
            else a = value_type(a_);
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return a;
        }


        //--------------------------------------------------------------------
        self_type& premultiply()
        {
            if (a < 0) v = 0;
            else if(a < 1) v *= a;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& demultiply()
        {
            if (a < 0) v = 0;
            else if (a < 1) v /= a;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            return self_type(
                             value_type(v + (c.v - v) * k),
                             value_type(a + (c.a - a) * k));
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };
}

#endif /* GRAY64_H */
