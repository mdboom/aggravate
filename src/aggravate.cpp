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

static PyObject *
Py_resample(PyObject *self, PyObject* args, PyObject *kwargs)
{
    int interpolation = NEAREST;
    PyObject *py_input_array = NULL;
    PyObject *py_output_array = NULL;
    double matrix[6];
    double norm = 0.0;
    double radius = 1.0;
    PyArrayObject *input_array = NULL;
    PyArrayObject *output_array = NULL;

    /* TODO: This could use the buffer interface to avoid the Numpy
       dependency, but this is easier for now. */

    const char *kwlist[] = {
        "input_array", "output_array",
        "sx", "shy", "shx", "sy", "tx", "ty",
        "interpolation", "norm", "radius", NULL };

    memset(matrix, 0, sizeof(double) * 6);

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOdddd|ddidd:resample", (char **)kwlist,
            &py_input_array, &py_output_array,
            &matrix[0], &matrix[1], &matrix[2], &matrix[3],
            &matrix[4], &matrix[5], &interpolation, &norm, &radius)) {
        return NULL;
    }

    if (interpolation < 0 || interpolation >= _n_interpolation) {
        PyErr_Format(PyExc_ValueError, "invalid interpolation value %d", interpolation);
        return NULL;
    }

    input_array = (PyArrayObject *)PyArray_ContiguousFromAny(py_input_array, NPY_DOUBLE, 2, 2);
    if (input_array == NULL) {
        return NULL;
    }

    output_array = (PyArrayObject *)PyArray_ContiguousFromAny(py_output_array, NPY_DOUBLE, 2, 2);
    if (output_array == NULL) {
        Py_DECREF(input_array);
        return NULL;
    }

    aggravate_resample(
        (interpolation_e)interpolation,
        (double *)PyArray_DATA(input_array),
        PyArray_DIM(input_array, 1),
        PyArray_DIM(input_array, 0),
        (double *)PyArray_DATA(output_array),
        PyArray_DIM(output_array, 1),
        PyArray_DIM(output_array, 0),
        matrix, norm, radius);

    Py_DECREF(input_array);
    return (PyObject *)output_array;
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
