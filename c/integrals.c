#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif


int dim3to1(int n1, int n2, int n3, int dim);

//static double FrequencyEvaluation(double Delta, double  Coefficients[], int m, double xms);
static PyObject* ExpIntegralSimple(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* ExpIntegral(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* PolyProduct(PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring_1[] =
    "integrate_exponential_simple(n, a)\n\n Solve integrals type: x^n exp(-ax^2)";
static char function_docstring_2[] =
    "integrate_exponential(n, a, b)\n\n Solve integrals type: x^n exp(-(ax^2+bx))";
static char function_docstring_3[] =
    "product_poly_coeff(poly_coeff, poly_coeff2, max_lim=None)\n\n Do product of two polynomials";


static PyMethodDef extension_funcs[] = {
    {"integrate_exponential_simple",  (PyCFunction)ExpIntegralSimple, METH_VARARGS|METH_KEYWORDS, function_docstring_1},
    {"integrate_exponential",  (PyCFunction)ExpIntegral, METH_VARARGS|METH_KEYWORDS, function_docstring_2},
    {"product_poly_coeff",  (PyCFunction)PolyProduct, METH_VARARGS|METH_KEYWORDS, function_docstring_3},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "integrals",
  "integrals module",
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
    m = Py_InitModule3("integrals",
        extension_funcs, "integrals module");
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
    PyInit_integrals(void)
    {
        import_array();
        return moduleinit();
    }

#endif

static int fact(int n)
{
    if (n == 0){
        return 1;
    }else{
        return(n * fact(n-1));
    }
}

static int nCr(int n, int r)
{
    return fact(n) / (fact(r) * fact(n - r));
}

static PyObject* ExpIntegralSimple(PyObject* self, PyObject *arg, PyObject *keywords)
{

    int n;
    double a;
    double PI = acos(-1.0);
    double b = 0.0;

    //  Interface with Python
    static char *kwlist[] = {"n", "a", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "id", kwlist, &n, &a))  return NULL;

    if (n == 0) {
        double integral = sqrt(PI/a)*exp(pow(b, 2)/(4.0*a));
        return Py_BuildValue("d", integral);
    }

    if (n == 1) {
        double integral = sqrt(PI)/(2.0*pow(a, 3.0/2))*b*exp(pow(b, 2)/(4.0*a));
        return Py_BuildValue("d", integral);
    }

    double factor = 0.0;

    for (int k = 0; k < n/2+1; k++)
    {
        factor += nCr(n, 2*k)*pow(b/(2.0*a), n-2*k) * fact(2*k)/(pow(2, 2*k)*fact(k)*pow(a,k));
    }

    //double factor = np.sum([math.comb(n, 2*k)*(b/(2*a))**(n-2*k)*math.factorial(2*k)/(2**(2*k)*math.factorial(k)*a**k)
    //                      for k in range(n//2+1)])

    double integral =  factor * sqrt(PI/a)*exp(pow(b,2)/(4*a));

    return Py_BuildValue("d", integral);
}



static PyObject* ExpIntegral(PyObject* self, PyObject *arg, PyObject *keywords)
{

    int n;
    double a, b;
    double integral;
    double PI = acos(-1.0);

    //  Interface with Python
    static char *kwlist[] = {"n", "a", "b", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "idd", kwlist, &n, &a, &b))  return NULL;

    //Create new numpy array for storing result

    if (n == 0) {
        integral = sqrt(PI/a)*exp(pow(b, 2)/(4.0*a));
        return Py_BuildValue("d", integral);
    }

    if (n == 1) {
        integral = sqrt(PI)/(2.0*pow(a, 3.0/2))*b*exp(pow(b, 2)/(4.0*a));
        return Py_BuildValue("d", integral);
    }

    double factor = 0.0;

    for (int k = 0; k < n/2+1; k++)
    {
        factor += nCr(n, 2*k)*pow(b/(2.0*a), n-2*k) * fact(2*k)/(pow(2, 2*k)*fact(k)*pow(a,k));
    }

    //double factor = np.sum([math.comb(n, 2*k)*(b/(2*a))**(n-2*k)*math.factorial(2*k)/(2**(2*k)*math.factorial(k)*a**k)
    //                      for k in range(n//2+1)])

    integral =  factor * sqrt(PI/a)*exp(pow(b,2)/(4*a));

    return Py_BuildValue("d", integral);
}

int dim3to1(int n1, int n2, int n3, int dim)
{
    return n1*dim*dim + n2*dim + n3;
}

static PyObject* PolyProduct(PyObject* self, PyObject *arg, PyObject *keywords)
{

    int maxLim;


    //  Interface with Python poly_coeff
    PyObject *polyCoeff_obj, *polyCoeff2_obj, *maxLim_obj;
    maxLim_obj = Py_None;

    static char *kwlist[] = {"polyCoeff", "polyCoeff2", "max_lim", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OO|O", kwlist, &polyCoeff_obj, &polyCoeff2_obj, &maxLim_obj))  return NULL;

    PyObject *polyCoeffArray = PyArray_FROM_OTF(polyCoeff_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *polyCoeff2Array = PyArray_FROM_OTF(polyCoeff2_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (polyCoeffArray == NULL || polyCoeff2Array == NULL ) {
        Py_XDECREF(polyCoeffArray);
        Py_XDECREF(polyCoeff2Array);
        return NULL;
    }

    double *polyCoeff    = (double*)PyArray_DATA(polyCoeffArray);
    double *polyCoeff2   = (double*)PyArray_DATA(polyCoeff2Array);

    int  maxLim1 = (int)PyArray_DIM(polyCoeffArray, 0);
    int  maxLim2 = (int)PyArray_DIM(polyCoeff2Array, 0);

    if (maxLim_obj == Py_None){
        maxLim = maxLim1 + maxLim2;
    }
    else{
        maxLim = (int) PyLong_AsLong(maxLim_obj);
    }

    //Create new numpy array for storing result
    PyArrayObject *polyCoeffArray_object;

    npy_intp dims[]={maxLim, maxLim, maxLim};
    polyCoeffArray_object=(PyArrayObject *) PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    double *polyCoeffProd  = (double*)PyArray_DATA(polyCoeffArray_object);

    for (int i = 0; i < pow(maxLim, 3); i++) polyCoeffProd[i] = 0.0;

    for (int i = 0; i < maxLim1; i++)
    {
        for (int j = 0; j < maxLim1; j++)
        {
            for (int k = 0; k < maxLim1; k++)
            {
                for (int i2 = 0; i2 < maxLim2; i2++)
                {
                    for (int j2 = 0; j2 < maxLim2; j2++)
                    {
                        for (int k2 = 0; k2 < maxLim2; k2++)
                        {
                            if (i + i2 < maxLim && j + j2 < maxLim && k + k2 < maxLim)
                            {
                            polyCoeffProd[dim3to1(i + i2, j + j2, k + k2, maxLim)] += polyCoeff[dim3to1(i, j, k, maxLim1)] * polyCoeff2[dim3to1(i2, j2, k2, maxLim2)];
                            }
                        }
                    }
                }

            }
        }
    }



    // polyCoeffProd[dim3to1(0, 0, 1, maxLim1)] = 8;

    // Free python memory
    Py_DECREF(polyCoeffArray);
    Py_DECREF(polyCoeff2Array);

    return(PyArray_Return(polyCoeffArray_object));


    //Create new numpy array for storing result
    //double f[5] = {0,1,2,3,4};
    //int d[1] = {5};
    //PyObject *c = PyArray_FromDims(1,d,NPY_DOUBLE);
    //memcpy(PyArray_DATA(c), f, 5*sizeof(double));
    //return PyArray_DATA(c);
}
