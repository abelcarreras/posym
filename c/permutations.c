#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

// Support functions
double evaluationFast(int i, int j, double* row1, double* row2);
void exchangeRows(int i, int j, double* matrix, int n, int m);
void exchangePos(int i, int j, int* vector);
void inversePerm(int* vector, int n);

//static double FrequencyEvaluation(double Delta, double  Coefficients[], int m, double xms);
static PyObject* getCrossDistanceTable(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* getPermutationSimple(PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring_1[] =
    "get_cross_distance_table(coordinates, operated_coor)\n\n get cross distance";
static char function_docstring_2[] =
    "get_permutation_simple(distance_table, symbols)\n\n get permutation";


static PyMethodDef extension_funcs[] = {
    {"get_cross_distance_table",  (PyCFunction)getCrossDistanceTable, METH_VARARGS|METH_KEYWORDS, function_docstring_1},
    {"get_permutation_simple",  (PyCFunction)getPermutationSimple, METH_VARARGS|METH_KEYWORDS, function_docstring_2},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "permutations",
  "permutations module",
  -1,
  extension_funcs,
  NULL,
  NULL,
  NULL,
  NULL,
};
#endif


static PyObject *moduleinit(void)
{
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("permutations",
        extension_funcs, "permutations module");
#endif

  return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initintegrals(void)
    {
        import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_permutations(void)
    {
        import_array();
        return moduleinit();
    }

#endif


static PyObject* getCrossDistanceTable(PyObject* self, PyObject *arg, PyObject *keywords)
{
    //  Interface with Python
    PyObject *coordinates_obj, *operatedCoor_obj;

    static char *kwlist[] = {"coordinates", "operated_coor", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OO", kwlist, &coordinates_obj, &operatedCoor_obj))  return NULL;

    PyObject *coordinatesArray = PyArray_FROM_OTF(coordinates_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *operatedCoorArray = PyArray_FROM_OTF(operatedCoor_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (coordinatesArray == NULL || operatedCoorArray == NULL ) {
        Py_XDECREF(coordinatesArray);
        Py_XDECREF(operatedCoorArray);
        return NULL;
    }

    double *coordinates    = (double*)PyArray_DATA(coordinatesArray);
    double *operatedCoor   = (double*)PyArray_DATA(operatedCoorArray);

    int  nCoor = (int)PyArray_DIM(coordinatesArray, 0);
    int  nCoor2 = (int)PyArray_DIM(operatedCoorArray, 0);
    int  nDim = (int)PyArray_DIM(coordinatesArray, 1);
    int  nDim2 = (int)PyArray_DIM(operatedCoorArray, 1);

    if (nDim != nDim2 || nCoor != nCoor2){
        Py_DECREF(coordinatesArray);
        Py_DECREF(operatedCoorArray);
        PyErr_SetString(PyExc_TypeError, "Coordinates do not have same dimensions");
        return (PyObject *) NULL;
     }

    //Create new numpy array for storing result
    PyArrayObject *diffCoor_obj;

    npy_intp dims[]={nCoor, nCoor2};
    diffCoor_obj=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *diffCoor = (double*)PyArray_DATA(diffCoor_obj);

    int ij, ik, jk;

    for (int i = 0; i < nCoor; i++) {
        for (int j = 0; j < nCoor2; j++) {
            ij = i*nCoor + j;
            diffCoor[ij] = 0.0;
            for (int k = 0; k < nDim; k++) {
                ik = i*nDim + k;
                jk = j*nDim + k;
                diffCoor[ij] += pow(coordinates[ik] - operatedCoor[jk], 2);
            }
            diffCoor[ij] = sqrt(diffCoor[ij]);
        }
    }

    // Free python memory
    Py_DECREF(coordinatesArray);
    Py_DECREF(operatedCoorArray);

    return(PyArray_Return(diffCoor_obj));
}

double evaluationFast(int i, int j, double* row1, double* row2)
{
        double off_diagonal = pow(row1[j] - row2[i], 2);
        return pow(row1[i], 2) + pow(row2[j], 2) + off_diagonal;
}

void exchangeRows(int i, int j, double* matrix, int n, int m)
{
        double temp;
        for (int k=0; k<m; k++){
            temp = matrix[i*n+k];
            matrix[i*n+k] = matrix[j*n+k];
            matrix[j*n+k] = temp;
        }
}

void exchangePos(int i, int j, int* vector)
{
        int temp;
        temp = vector[i];
        vector[i] = vector[j];
        vector[j] = temp;
}


void inversePerm(int* vector, int n)
{

    int * vectorTemp = (int*) malloc(n * sizeof(int));

    int k;
    for (int i=0; i<n; i++){
        k = vector[i];
        vectorTemp[k] = i;
    }
    for (int i=0; i<n; i++) vector[i] = vectorTemp[i];
    free(vectorTemp);

}


static PyObject* getPermutationSimple(PyObject* self, PyObject *arg, PyObject *keywords)
{

    //  Interface with Python
    PyObject *distanceTable_obj, *symbols_obj;

    static char *kwlist[] = {"distance_table", "symbols", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OO", kwlist, &distanceTable_obj, &symbols_obj))  return NULL;

    PyObject *distanceTableArray = PyArray_FROM_OTF(distanceTable_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *symbolsArray = PyArray_FROM_OTF(symbols_obj, NPY_INT, NPY_IN_ARRAY);

    if (distanceTableArray == NULL ) {
        Py_XDECREF(distanceTableArray);
        Py_XDECREF(symbolsArray);
        return NULL;
    }

    double *distanceTable = (double*)PyArray_DATA(distanceTableArray);
    int* symbols = (int*)PyArray_DATA(symbolsArray);

    int  nCoor1 = (int)PyArray_DIM(distanceTableArray, 0);
    int  nCoor2 = (int)PyArray_DIM(distanceTableArray, 1);
    int  n = (int)PyArray_DIM(symbolsArray, 0);

    if (n != nCoor1 || n != nCoor2){
        Py_DECREF(distanceTableArray);
        Py_DECREF(symbolsArray);
        PyErr_SetString(PyExc_TypeError, "Dimensions error ");
        return (PyObject *) NULL;
    }

    PyArrayObject *perm_obj;
    npy_intp dims[]={n};
    perm_obj=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);
    int *perm = (int*)PyArray_DATA(perm_obj);

    for (int i=0; i<n; i++)perm[i] = i;

    double *row_1;
    double *row_2;

    int j = 0;
    int control = 1;
    while (control){
        control = 0;
        for (int a=n-1; a>1; a-- ){
            for (int i=0; i<n; i++ ){
                if (i+a < n){
                    j = i+a;
                } else {
                    j = i+a - n;
                }

                row_1 = distanceTable + i*n;
                row_2 = distanceTable + j*n;
                if (symbols[i] == symbols[j]) {
                    if (evaluationFast(i, j, row_1, row_2) > evaluationFast(j, i, row_1, row_2)){
                        exchangeRows(i, j, distanceTable, n, n);
                        exchangePos(i, j, perm);
                        control = 1;
                    }

                }

            }

        }

    }

    inversePerm(perm, n);

    // Free python memory
    Py_DECREF(distanceTableArray);
    Py_DECREF(symbolsArray);

    return(PyArray_Return(perm_obj));

}

