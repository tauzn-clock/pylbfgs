#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <lbfgs.h>


static char module_docstring[] =
    "This module provides a limited interface to the Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) algorithm written in C.";
static char owlqs_docstring[] =
    "Calculate the optimum of an objective of the least-squares form plus the L1-norm of the parameters.";

static PyObject *pyelwon_owlqn(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"owlqn", pyelwon_owlqn, METH_VARARGS, owlqs_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit_pyelwon(void)
{
    
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "pyelwon",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    /* Load `numpy` functionality. */
    import_array();

    return module;
}


PyObject *a_matrix, *a_matrix_transpose, *b_vector;

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    /* Calculate the 2-norm squared of the residual vector */
    lbfgsfloatval_t fx = 0.0;
    npy_intp dims[2] = {n, 1};
    PyObject *x_vector = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)x);
    PyObject *p_vector = PyArray_MatrixProduct(a_matrix, x_vector);
    lbfgsfloatval_t *p_data = (lbfgsfloatval_t*)PyArray_DATA(p_vector);
    lbfgsfloatval_t *b_data = (lbfgsfloatval_t*)PyArray_DATA(b_vector);
    int i;
    int mb = (int)PyArray_DIM(b_vector, 0);
    for (i = 0; i < mb; i++) {
        fx += pow(b_data[i] - p_data[i], 2);
    }

    /* Calculate the gradient vector */
    PyObject *atax_vector = PyArray_MatrixProduct(a_matrix_transpose, p_vector);
    PyObject *atb_vector = PyArray_MatrixProduct(a_matrix_transpose, b_vector);
    lbfgsfloatval_t *atax_data = (lbfgsfloatval_t*)PyArray_DATA(atax_vector);
    lbfgsfloatval_t *atb_data = (lbfgsfloatval_t*)PyArray_DATA(atb_vector);
    for(i = 0; i < n; i++) {
        g[i] = 2 * (atax_data[i] - atb_data[i]);
    }

    /* Clean up. */
    Py_DECREF(x_vector);
    Py_DECREF(p_vector);
    Py_DECREF(atax_vector);
    Py_DECREF(atb_vector);

    return fx;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    PySys_WriteStdout("Iteration %d:\n", k);
    PySys_WriteStdout("  fx = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm, step);
    PySys_WriteStdout("\n");
    return 0;
}


static PyObject *pyelwon_owlqn(PyObject *self, PyObject *args)
{
    double param_c;
    PyObject *a_obj, *b_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOd", &a_obj, &b_obj, &param_c))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    a_matrix = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    b_vector = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (a_matrix == NULL || b_vector == NULL) {
        Py_XDECREF(a_matrix);
        Py_XDECREF(b_vector);
        return NULL;
    }

    /* Get matrix dimensions */
    int ndima = PyArray_NDIM(a_matrix);
    int ma = (int)PyArray_DIM(a_matrix, 0);
    int na = (int)PyArray_DIM(a_matrix, 1);
    int ndimb = PyArray_NDIM(b_vector);
    int mb = (int)PyArray_DIM(b_vector, 0);
    int nb = (int)PyArray_DIM(b_vector, 1);
    // printf("ndima = %d\n", ndima);
    // printf("ndimb = %d\n", ndimb);

    /* Assert valid matrix dims */
    if (ndima != 2 || ndimb != 2) {
        Py_DECREF(a_matrix);
        Py_DECREF(b_vector);
        PyErr_SetString(PyExc_ValueError, "Matrices must have 2-dimensions.");
        return NULL;
    }
    if (ma != mb) {
        Py_DECREF(a_matrix);
        Py_DECREF(b_vector);
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions do not agree.");
        return NULL;
    }
    if (nb != 1) {
        Py_DECREF(a_matrix);
        Py_DECREF(b_vector);
        PyErr_SetString(PyExc_ValueError, "B must be a column vector.");
        return NULL;
    }

    /* Calculate transpose of A */
    a_matrix_transpose = PyArray_Transpose((PyArrayObject*)a_matrix, NULL);

    /* Initialize solution vector */
    int i;
    int n = na;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(n);
    if (x == NULL) {
        PySys_WriteStdout("ERROR: Failed to allocate a memory block for variables.\n");
        return NULL;
    }
    for (i = 0; i < n; i++) {
        x[i] = 1;
    }

    /* Initialize the parameters for the optimization. */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (lbfgsfloatval_t)param_c; // this tells lbfgs to do OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    // printf("params.c = %f\n", param.orthantwise_c);
    // printf("%d\n", LBFGSERR_INVALID_LINESEARCH);
    int lbfgs_ret = lbfgs(n, x, &fx, evaluate, progress, NULL, &param);
    PySys_WriteStdout("OWL-QN optimization terminated with status code = %d\n", lbfgs_ret);

    /* Copy solution to numpy array and free x */
    npy_intp dims[2] = {n, 1};
    PyObject *x_vector = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *xptr;
    for (i = 0; i < n; i++) {
        xptr = PyArray_GETPTR2(x_vector, i, 0);
        *xptr = (double)x[i];
    }
    lbfgs_free(x);

    /* Clean up. */
    Py_DECREF(a_matrix);
    Py_DECREF(b_vector);
    Py_DECREF(a_matrix_transpose);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("N", x_vector);
    return ret;
}