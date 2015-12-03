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

#include "aggravate.h"

#include "resample.h"


template<class T>
void aggravate_resample_allow_threads(
    interpolation_e interpolation,
    T *input, int in_width, int in_height,
    T *output, int out_width, int out_height,
    double *matrix, double norm, double radius)
{
    Py_BEGIN_ALLOW_THREADS
    aggravate_resample_parallel(
        interpolation,
        input, in_width, in_height,
        output, out_width, out_height,
        matrix, norm, radius);
    Py_END_ALLOW_THREADS
}


static PyObject *
Py_resample(PyObject *self, PyObject* args, PyObject *kwargs)
{
    int interpolation = NEAREST;
    PyObject *py_input_array = NULL;
    PyObject *py_output_array = NULL;
    PyObject *py_matrix_array = NULL;
    double norm = 0.0;
    double radius = 1.0;
    PyArrayObject *input_array = NULL;
    PyArrayObject *output_array = NULL;
    PyArrayObject *matrix_array = NULL;

    /* TODO: This could use the buffer interface to avoid the Numpy
       dependency, but this is easier for now. */

    const char *kwlist[] = {
        "input_array", "output_array",
        "matrix", "interpolation", "norm", "radius", NULL };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|idd:resample", (char **)kwlist,
            &py_input_array, &py_output_array, &py_matrix_array,
            &interpolation, &norm, &radius)) {
        return NULL;
    }

    if (interpolation < 0 || interpolation >= _n_interpolation) {
        PyErr_Format(PyExc_ValueError, "invalid interpolation value %d", interpolation);
        goto error;
    }

    input_array = (PyArrayObject *)PyArray_FromAny(
        py_input_array, NULL, 2, 3, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (input_array == NULL) {
        goto error;
    }

    output_array = (PyArrayObject *)PyArray_FromAny(
        py_output_array, NULL, 2, 3, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (output_array == NULL) {
        goto error;
    }

    matrix_array = (PyArrayObject *)PyArray_ContiguousFromAny(
        py_matrix_array, NPY_DOUBLE, 2, 2);
    if (matrix_array == NULL) {
        goto error;
    }

    if (PyArray_DIM(matrix_array, 0) != 3 ||
        PyArray_DIM(matrix_array, 1) != 3) {
        PyErr_Format(
            PyExc_ValueError, "Matrix must be 3x3, got %dx%d",
            PyArray_DIM(matrix_array, 0), PyArray_DIM(matrix_array, 1));
        goto error;
    }

    if (PyArray_NDIM(input_array) != PyArray_NDIM(output_array)) {
        PyErr_Format(PyExc_ValueError, "Mismatched number of dimensions.  Got %d and %d.",
                     PyArray_NDIM(input_array), PyArray_NDIM(output_array));
        goto error;
    }

    if (PyArray_TYPE(input_array) != PyArray_TYPE(output_array)) {
        PyErr_SetString(PyExc_ValueError, "Mismatched types.");
        goto error;
    }

    if (PyArray_NDIM(input_array) == 3) {
        if (PyArray_TYPE(input_array) != NPY_UBYTE ||
            PyArray_TYPE(input_array) != NPY_UINT8) {
            PyErr_SetString(PyExc_ValueError,
                            "3-dimensional arrays must be of type unsigned byte");
            goto error;
        }

        if (PyArray_DIM(input_array, 2) != PyArray_DIM(output_array, 2)) {
            PyErr_SetString(PyExc_ValueError,
                            "Mismatched shapes");
            goto error;
        }

        // if (PyArray_DIM(input_array, 2) == 3) {
        //     aggravate_resample(
        //         (interpolation_e)interpolation,
        //         (agg::rgb8 *)PyArray_DATA(input_array),
        //         PyArray_DIM(input_array, 1),
        //         PyArray_DIM(input_array, 0),
        //         (agg::rgb8 *)PyArray_DATA(output_array),
        //         PyArray_DIM(output_array, 1),
        //         PyArray_DIM(output_array, 0),
        //         matrix, norm, radius);
        if (PyArray_DIM(input_array, 2) == 4) {
            aggravate_resample_allow_threads(
                (interpolation_e)interpolation,
                (agg::rgba8 *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (agg::rgba8 *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                (double *)PyArray_DATA(matrix_array),
                norm, radius);
        } else {
            PyErr_Format(PyExc_ValueError,
                         "If 3-dimensional, array must be RGBA.  Got %d.",
                         PyArray_DIM(input_array, 2));
            goto error;
        }
    } else { // NDIM == 2
        if (PyArray_TYPE(input_array) == NPY_DOUBLE) {
            aggravate_resample_allow_threads(
                (interpolation_e)interpolation,
                (double *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (double *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                (double *)PyArray_DATA(matrix_array),
                norm, radius);
        } else if (PyArray_TYPE(input_array) == NPY_FLOAT) {
            aggravate_resample_allow_threads(
                (interpolation_e)interpolation,
                (float *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (float *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                (double *)PyArray_DATA(matrix_array),
                norm, radius);
        } else if (PyArray_TYPE(input_array) == NPY_UINT8) {
            aggravate_resample_allow_threads(
                (interpolation_e)interpolation,
                (unsigned char *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (unsigned char *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                (double *)PyArray_DATA(matrix_array),
                norm, radius);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
            goto error;
        }
    }

    Py_DECREF(input_array);
    Py_DECREF(matrix_array);
    return (PyObject *)output_array;

 error:
    Py_XDECREF(input_array);
    Py_XDECREF(output_array);
    Py_XDECREF(matrix_array);
    return NULL;
}


static PyMethodDef module_methods[] = {
    {"resample", (PyCFunction)Py_resample, METH_VARARGS|METH_KEYWORDS, "resample"},
    {NULL}  /* Sentinel */
};


PyObject *m;


/* TODO: Hide all exported symbols in the shared library except this one */

#if PY3K
    static void
    aggravate_module_dealloc(PyObject *m)
    {

    }

    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_aggravate",
        NULL,
        0,
        module_methods,
        NULL,
        NULL,
        NULL,
        (freefunc)aggravate_module_dealloc
    };

    #define INITERROR return NULL

    PyMODINIT_FUNC
    PyInit__aggravate(void)
#else
    #define INITERROR return

    PyMODINIT_FUNC
    init_aggravate(void)
#endif
{

#if PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3(
        "_aggravate", module_methods,
        "2D image interpolation library");
#endif

    if (m == NULL) {
        INITERROR;
    }

    import_array();

    if (PyModule_AddIntConstant(m, "NEAREST", NEAREST) ||
        PyModule_AddIntConstant(m, "BILINEAR", BILINEAR) ||
        PyModule_AddIntConstant(m, "BICUBIC", BICUBIC) ||
        PyModule_AddIntConstant(m, "SPLINE16", SPLINE16) ||
        PyModule_AddIntConstant(m, "SPLINE36", SPLINE36) ||
        PyModule_AddIntConstant(m, "HANNING", HANNING) ||
        PyModule_AddIntConstant(m, "HAMMING", HAMMING) ||
        PyModule_AddIntConstant(m, "HERMITE", HERMITE) ||
        PyModule_AddIntConstant(m, "KAISER", KAISER) ||
        PyModule_AddIntConstant(m, "QUADRIC", QUADRIC) ||
        PyModule_AddIntConstant(m, "CATROM", CATROM) ||
        PyModule_AddIntConstant(m, "GAUSSIAN", GAUSSIAN) ||
        PyModule_AddIntConstant(m, "BESSEL", BESSEL) ||
        PyModule_AddIntConstant(m, "MITCHELL", MITCHELL) ||
        PyModule_AddIntConstant(m, "SINC", SINC) ||
        PyModule_AddIntConstant(m, "LANCZOS", LANCZOS) ||
        PyModule_AddIntConstant(m, "BLACKMAN", BLACKMAN) ||
        PyModule_AddIntConstant(m, "_n_interpolation", _n_interpolation)) {
        INITERROR;
    }

    #if PY3K
    return m;
    #endif
}
