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

#ifndef RESAMPLE_H
#define RESAMPLE_H

#include "agg_image_accessors.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_renderer_base.h"
#include "agg_renderer_scanline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_scanline_u.h"
#include "agg_span_allocator.h"
#include "agg_span_image_filter_gray.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_linear.h"

#include "gray64.h"


typedef enum {
    NEAREST,
    BILINEAR,
    BICUBIC,
    SPLINE16,
    SPLINE36,
    HANNING,
    HAMMING,
    HERMITE,
    KAISER,
    QUADRIC,
    CATROM,
    GAUSSIAN,
    BESSEL,
    MITCHELL,
    SINC,
    LANCZOS,
    BLACKMAN,
    _n_interpolation
} interpolation_e;


template <typename T>
struct type_mapping;


template <> class type_mapping<double>
{
 public:
    typedef agg::gray64 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;

    template <typename A>
    struct span_gen_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};


template <> class type_mapping<agg::rgba8>
{
 public:
    typedef agg::rgba8 color_type;
    typedef agg::blender_rgba32 blender_type;
    typedef agg::pixfmt_alpha_blend_rgba<blender_type, agg::rendering_buffer> pixfmt_type;

    template <typename A>
    struct span_gen_type
    {
        typedef agg::span_image_resample_rgba_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_rgba_nn<A, B> type;
    };
};


template <> class type_mapping<float>
{
 public:
    typedef agg::gray32 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;

    template <typename A>
    struct span_gen_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};


template <> class type_mapping<unsigned char>
{
 public:
    typedef agg::gray8 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;

    template <typename A>
    struct span_gen_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};



template<class T>
void aggravate_resample(
    interpolation_e interpolation,
    T *input, int in_width, int in_height,
    T *output, int out_width, int out_height,
    double *matrix, double norm, double radius,
    int row_offset)
{
    typedef type_mapping<T> type_mapping_t;

    typedef typename type_mapping_t::pixfmt_type input_pixfmt_t;
    typedef typename type_mapping_t::pixfmt_type output_pixfmt_t;

    typedef agg::renderer_base<output_pixfmt_t> renderer_t;
    typedef agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl> rasterizer_t;

    typedef agg::wrap_mode_reflect reflect_t;
    typedef agg::image_accessor_wrap<input_pixfmt_t, reflect_t, reflect_t> image_accessor_t;

    typedef agg::span_allocator<typename type_mapping_t::color_type> span_alloc_t;
    typedef agg::span_interpolator_linear<> interpolator_t;

    agg::trans_affine affine(
        matrix[0], matrix[3], matrix[1], matrix[4], matrix[2],
        matrix[5] - row_offset);
    agg::trans_affine inverted = affine;
    inverted.invert();

    agg::rendering_buffer input_buffer;
    input_buffer.attach((unsigned char *)input, in_width, in_height,
                        in_width * sizeof(T));

    input_pixfmt_t input_pixfmt(input_buffer);
    image_accessor_t input_accessor(input_pixfmt);
    span_alloc_t span_alloc;

    agg::rendering_buffer output_buffer;
    output_buffer.attach((unsigned char *)output, out_width, out_height,
                         out_width * sizeof(T));

    output_pixfmt_t output_pixfmt(output_buffer);
    renderer_t renderer(output_pixfmt);
    rasterizer_t rasterizer;
    agg::scanline_u8 scanline;

    rasterizer.clip_box(0, 0, out_width, out_height);

    agg::path_storage path;
    path.move_to(0, 0);
    path.line_to(in_width, 0);
    path.line_to(in_width, in_height);
    path.line_to(0, in_height);
    path.close_polygon();
    agg::conv_transform<agg::path_storage> rectangle(path, affine);

    rasterizer.add_path(rectangle);

    interpolator_t interpolator(inverted);

    if (interpolation == NEAREST) {
        typedef typename type_mapping_t::template span_gen_nn_type<image_accessor_t, interpolator_t>::type span_gen_t;
        typedef agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_gen_t> nn_renderer_t;

        span_gen_t span_gen(input_accessor, interpolator);
        nn_renderer_t nn_renderer(renderer, span_alloc, span_gen);
        agg::render_scanlines(rasterizer, scanline, nn_renderer);

    } else {
        agg::image_filter_lut filter;
        switch (interpolation) {
        case NEAREST:
        case _n_interpolation:
            // Never should get here.  Here to silence compiler warnings.
            break;

        case HANNING:
            filter.calculate(agg::image_filter_hanning(), norm);
            break;

        case HAMMING:
            filter.calculate(agg::image_filter_hamming(), norm);
            break;

        case HERMITE:
            filter.calculate(agg::image_filter_hermite(), norm);
            break;

        case BILINEAR:
            filter.calculate(agg::image_filter_bilinear(), norm);
            break;

        case BICUBIC:
            filter.calculate(agg::image_filter_bicubic(), norm);
            break;

        case SPLINE16:
            filter.calculate(agg::image_filter_spline16(), norm);
            break;

        case SPLINE36:
            filter.calculate(agg::image_filter_spline36(), norm);
            break;

        case KAISER:
            filter.calculate(agg::image_filter_kaiser(), norm);
            break;

        case QUADRIC:
            filter.calculate(agg::image_filter_quadric(), norm);
            break;

        case CATROM:
            filter.calculate(agg::image_filter_catrom(), norm);
            break;

        case GAUSSIAN:
            filter.calculate(agg::image_filter_gaussian(), norm);
            break;

        case BESSEL:
            filter.calculate(agg::image_filter_bessel(), norm);
            break;

        case MITCHELL:
            filter.calculate(agg::image_filter_mitchell(), norm);
            break;

        case SINC:
            filter.calculate(agg::image_filter_sinc(radius), norm);
            break;

        case LANCZOS:
            filter.calculate(agg::image_filter_lanczos(radius), norm);
            break;

        case BLACKMAN:
            filter.calculate(agg::image_filter_blackman(radius), norm);
            break;
        }

        typedef typename type_mapping_t::template span_gen_type<image_accessor_t>::type span_gen_t;
        typedef agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_gen_t> int_renderer_t;

        span_gen_t span_gen(input_accessor, interpolator, filter);
        int_renderer_t int_renderer(renderer, span_alloc, span_gen);
        agg::render_scanlines(rasterizer, scanline, int_renderer);
    }
}


template<class T>
void aggravate_resample_parallel(
    interpolation_e interpolation,
    T *input, int in_width, int in_height,
    T *output, int out_width, int out_height,
    double *matrix, double norm, double radius)
{
    int i;
    int step_size = 256;

    #pragma omp parallel for
    for (i = 0; i < out_height; i += step_size) {
        aggravate_resample(
            interpolation,
            input, in_width, in_height,
            output + (i * out_width), out_width,
            std::min(out_height - i, step_size),
            matrix, norm, radius, i);
    }
}


#endif /* RESAMPLE_H */
