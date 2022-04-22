#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif


// Support functions
int dim3to1(int n1, int n2, int n3, int dim);
double ExpIntegralC(int n, double a, double b);

//static double FrequencyEvaluation(double Delta, double  Coefficients[], int m, double xms);
static PyObject* ExpIntegralSimple(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* ExpIntegral(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* PolyProduct(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* GaussianIntegral(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* GaussianIntegral2(PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring_1[] =
    "integrate_exponential_simple(n, a)\n\n Solve integrals type: x^n exp(-ax^2)";
static char function_docstring_2[] =
    "integrate_exponential(n, a, b)\n\n Solve integrals type: x^n exp(-(ax^2+bx))";
static char function_docstring_3[] =
    "product_poly_coeff(poly_coeff, poly_coeff2, max_lim=None)\n\n Do product of two polynomials";
static char function_docstring_4[] =
    "gaussian_integral(poly_coeff, )\n\n Solve gaussian integral x^n y^m z^l exp(-a(r-R)^2)";


static PyMethodDef extension_funcs[] = {
    {"integrate_exponential_simple",  (PyCFunction)ExpIntegralSimple, METH_VARARGS|METH_KEYWORDS, function_docstring_1},
    {"integrate_exponential",  (PyCFunction)ExpIntegral, METH_VARARGS|METH_KEYWORDS, function_docstring_2},
    {"product_poly_coeff",  (PyCFunction)PolyProduct, METH_VARARGS|METH_KEYWORDS, function_docstring_3},
    {"gaussian_integral",  (PyCFunction)GaussianIntegral, METH_VARARGS|METH_KEYWORDS, function_docstring_4},
    {"gaussian_integral_2",  (PyCFunction)GaussianIntegral2, METH_VARARGS|METH_KEYWORDS, function_docstring_4},

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

    double integral =  factor * sqrt(PI/a)*exp(pow(b,2)/(4*a));

    return Py_BuildValue("d", integral);
}


double ExpIntegralC(int n, double a, double b)
{

    double integral;
    double PI = acos(-1.0);


    if (n == 0) {
        integral = sqrt(PI/a)*exp(pow(b, 2)/(4.0*a));
        return integral;
    }

    if (n == 1) {
        integral = sqrt(PI)/(2.0*pow(a, 3.0/2))*b*exp(pow(b, 2)/(4.0*a));
        return integral;
    }

    double factor = 0.0;

    for (int k = 0; k < n/2+1; k++)
    {
        factor += nCr(n, 2*k)*pow(b/(2.0*a), n-2*k) * fact(2*k)/(pow(2, 2*k)*fact(k)*pow(a,k));
    }

    integral =  factor * sqrt(PI/a)*exp(pow(b,2)/(4*a));

    return integral;
}


static PyObject* ExpIntegral(PyObject* self, PyObject *arg, PyObject *keywords)
{

    int n;
    double a, b;
    // double integral;
    // double PI = acos(-1.0);

    // Interface with Python
    static char *kwlist[] = {"n", "a", "b", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "idd", kwlist, &n, &a, &b))  return NULL;

    // Create new numpy array for storing result
    return Py_BuildValue("d", ExpIntegralC(n, a, b));

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


    // Free python memory
    Py_DECREF(polyCoeffArray);
    Py_DECREF(polyCoeff2Array);

    return(PyArray_Return(polyCoeffArray_object));

}


static PyObject* GaussianIntegral(PyObject* self, PyObject *arg, PyObject *keywords)
{

    //  Interface with Python
    PyObject *polyCoeff_obj, *center_obj;
    double alpha;

    static char *kwlist[] = {"alpha", "center", "poly_coeff", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "dOO", kwlist, &alpha, &center_obj, &polyCoeff_obj))  return NULL;

    PyObject *polyCoeffArray = PyArray_FROM_OTF(polyCoeff_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *centerArray = PyArray_FROM_OTF(center_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (polyCoeffArray == NULL || centerArray == NULL ) {
        Py_XDECREF(polyCoeffArray);
        Py_XDECREF(centerArray);
        return NULL;
    }

    double *polyCoeff    = (double*)PyArray_DATA(polyCoeffArray);
    double *center   = (double*)PyArray_DATA(centerArray);

    int maxLim = (int)PyArray_DIM(polyCoeffArray, 0);

    // Dot product center
    double dot_center = 0.0;
    for (int i = 0; i < 3; i++) {
        dot_center += center[i] * center[i];
    }

    double pre_exponential = exp(-alpha * dot_center);

    double integral = 0.0;
//    omp_get_max_threads(4);
//    # pragma omp parallel for reduction(+:integral) default(shared)
    for (int i = 0; i < maxLim; i++) {
        for (int j = 0; j < maxLim; j++) {
            for (int k = 0; k < maxLim; k++) {
                //printf("coef: %f\n", polyCoeff[dim3to1(i, j, k, maxLim)]);
                integral += polyCoeff[dim3to1(i, j, k, maxLim)] *
                    ExpIntegralC(i, alpha, 2*alpha*center[0]) *
                    ExpIntegralC(j, alpha, 2*alpha*center[1]) *
                    ExpIntegralC(k, alpha, 2*alpha*center[2]);
            }
        }
    }

    // Free python memory
    Py_DECREF(polyCoeffArray);

    return Py_BuildValue("d", integral * pre_exponential);
}

struct expContainer {
    double prefactor, exponent;
};


struct expContainer ExpIntegralPartial(int n, double a, double b)
{

    double PI = acos(-1.0);

    struct expContainer integral;

    if (n == 0) {
        integral.prefactor = sqrt(PI/a);
        integral.exponent = b*b/(4.0*a);
        return integral;
    }

    if (n == 1) {
        integral.prefactor = sqrt(PI)/(2.0*pow(a, 3.0/2.0))*b;
        integral.exponent = b*b/(4.0*a);

        return integral;
    }

    double factor = 0.0;

    for (int k = 0; k < n/2+1; k++){
        factor += nCr(n, 2*k)*pow(b/(2.0*a), n-2*k) * fact(2*k)/(pow(2, 2*k)*fact(k)*pow(a,k));
    }

    integral.prefactor = factor * sqrt(PI/a);
    integral.exponent = pow(b,2)/(4*a);
    return integral;
}

int getMinFomList(double *list, int n){
    double min = list[0];
    int index = 0;

    for (int i=1; i<n; i++){
        if (list[i] < min){
            index = i;
            min = list[i];
        }
    }

    return index;

}

static PyObject* GaussianIntegral2(PyObject* self, PyObject *arg, PyObject *keywords)
{

    //  Interface with Python
    PyObject *polyCoeff_obj, *center_obj;
    double alpha;

    static char *kwlist[] = {"alpha", "center", "poly_coeff", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "dOO", kwlist, &alpha, &center_obj, &polyCoeff_obj))  return NULL;

    PyObject *polyCoeffArray = PyArray_FROM_OTF(polyCoeff_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *centerArray = PyArray_FROM_OTF(center_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (polyCoeffArray == NULL || centerArray == NULL ) {
        Py_XDECREF(polyCoeffArray);
        Py_XDECREF(centerArray);
        return NULL;
    }

    double *polyCoeff    = (double*)PyArray_DATA(polyCoeffArray);
    double *center   = (double*)PyArray_DATA(centerArray);

    int maxLim = (int)PyArray_DIM(polyCoeffArray, 0);
    int n;

    struct expContainer intX, intY, intZ;
    // Dot product center
    double dot_center = 0.0;
    for (int i = 0; i < 3; i++) {
        dot_center += center[i] * center[i];
    }

    double generalExponent = -alpha * dot_center;

    //double integral = 0.0;
    int totalDim = maxLim*maxLim*maxLim;
//    omp_get_max_threads(4);
//    # pragma omp parallel for reduction(+:integral) default(shared)
    double * expList = (double*) malloc(totalDim * sizeof(double));
    double * preExpList = (double*) malloc(totalDim * sizeof(double));
    for (int i = 0; i < maxLim; i++) {
        for (int j = 0; j < maxLim; j++) {
            for (int k = 0; k < maxLim; k++) {
                n = i*maxLim*maxLim + j * maxLim + k;
                intX = ExpIntegralPartial(i, alpha, 2*alpha*center[0]);
                intY = ExpIntegralPartial(j, alpha, 2*alpha*center[1]);
                intZ = ExpIntegralPartial(k, alpha, 2*alpha*center[2]);
                expList[n] = intX.exponent + intY.exponent + intZ.exponent;
                preExpList[n] = polyCoeff[dim3to1(i, j, k, maxLim)] * intX.prefactor * intY.prefactor * intZ.prefactor;
            }
        }
    }

    int lowIndex = getMinFomList(expList, totalDim);
    double commonExp = expList[lowIndex];
    for (int i = 0; i < totalDim; i++) {
        expList[i] -= commonExp;
    }

    double preExponential = 0.0;
    for (int i = 0; i < totalDim; i++) {
        preExponential += preExpList[i]*exp(expList[i]);
    }

    // Free python memory
    Py_DECREF(polyCoeffArray);
    free(expList);
    free(preExpList);

    return Py_BuildValue("d", preExponential * exp(commonExp + generalExponent));
}
