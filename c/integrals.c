#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif


//static double FrequencyEvaluation(double Delta, double  Coefficients[], int m, double xms);
static PyObject* ExpIntegralSimple(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* ExpIntegral(PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring_1[] =
    "integrate_exponential_simple(n, a)\n\n Solve integrals type: x^n exp(-ax^2)";
static char function_docstring_2[] =
    "integrate_exponential(n, a, b)\n\n Solve integrals type: x^n exp(-(ax^2+bx))";


static PyMethodDef extension_funcs[] = {
    {"integrate_exponential_simple",  (PyCFunction)ExpIntegralSimple, METH_VARARGS|METH_KEYWORDS, function_docstring_1},
    {"integrate_exponential",  (PyCFunction)ExpIntegral, METH_VARARGS|METH_KEYWORDS, function_docstring_2},
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

