#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include <time.h>

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
static PyObject* getPermutationAnnealing(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* getPermutationBruteForce(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* fixPermutationSimple(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* validatePermutation(PyObject* self, PyObject *arg, PyObject *keywords);

//  Python Interface
static char function_docstring_1[] =
    "get_cross_distance_table(coordinates, operated_coor)\n\n get cross distance";
static char function_docstring_2[] =
    "get_permutation_simple(distance_table, symbols)\n\n get permutation";
static char function_docstring_3[] =
    "get_permutation_annealing(distance_table, symbols)\n\n get permutation";
static char function_docstring_4[] =
    "get_permutation_brute(distance_table, symbols)\n\n get permutation";
static char function_docstring_5[] =
    "fix_permutation(distance_table, permutation, symbols)\n\n get permutation";
static char function_docstring_6[] =
    "validate_permutation(permutation, order, determinant)\n\n check is permutation is valid";


static PyMethodDef extension_funcs[] = {
    {"get_cross_distance_table",  (PyCFunction)getCrossDistanceTable, METH_VARARGS|METH_KEYWORDS, function_docstring_1},
    {"get_permutation_simple",  (PyCFunction)getPermutationSimple, METH_VARARGS|METH_KEYWORDS, function_docstring_2},
    {"get_permutation_annealing", (PyCFunction)getPermutationAnnealing, METH_VARARGS|METH_KEYWORDS, function_docstring_3},
    {"get_permutation_brute", (PyCFunction)getPermutationBruteForce, METH_VARARGS|METH_KEYWORDS, function_docstring_4},
    {"fix_permutation", (PyCFunction)fixPermutationSimple, METH_VARARGS|METH_KEYWORDS, function_docstring_5},
    {"validate_permutation", (PyCFunction)validatePermutation, METH_VARARGS|METH_KEYWORDS, function_docstring_6},
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

// Define structure type to contain all orbits data
typedef struct orbitsData {
  int *orbitsMatrix;
  int *orbitsLen;
  int numRows;
  int numColumns;
  int exponent;
} OrbitsData;


static int two2one(int row, int column, int numColumns) {
	return row * numColumns + column;
}


void printMatrix(OrbitsData orbits) {
    printf("\nPrint matrix: %i, %i\n", orbits.numRows, orbits.numColumns);
    for (int i=0; i < orbits.numRows; i++){
        if (orbits.orbitsLen[i] == 0) break;
        for (int j=0; j< orbits.orbitsLen[i]; j++) printf("%i ", orbits.orbitsMatrix[two2one(i, j, orbits.numColumns)]);
        printf(" : %i\n", orbits.orbitsLen[i]);
    }
}


int ascend(void const *a, void const *b)
{
    if ( *(int *)a > *(int *)b ) return 1;
    return -1;
}

void rollOrbit(int* orbit, int size, int nRoll){

    int iOrigin;
    int iTarget = 0;
    int backupFirst = orbit[0];

    for (int j=0; j<size-1; j++){
        iOrigin = iTarget - nRoll;
        while (iOrigin >= size) iOrigin -= size;
        while (iOrigin < 0) iOrigin += size;
        orbit[iTarget] = orbit[iOrigin];
        iTarget = iOrigin;
    }
    orbit[iTarget] = backupFirst;

}


void orbit2perm(int* perm, OrbitsData orbits){


    // get orbits data
    int* matrix = orbits.orbitsMatrix;
    int* lenMatrix = orbits.orbitsLen;
    int numRows = orbits.numRows;
    int numColumns = orbits.numColumns;

    int *orbit = malloc(sizeof(int) * numColumns);
    int *orbitRef = malloc(sizeof(int) * numColumns);


    // Loop for all orbits list
    for (int i=0; i < numRows; i++){

        if (lenMatrix[i] == 0) break;

        // get orbit
        for (int j=0; j < lenMatrix[i]; j++){
            orbit[j] =  matrix[two2one(i, j, numColumns)];
            orbitRef[j] = orbit[j];
        }

        // roll orbit reference
        rollOrbit(orbitRef, lenMatrix[i], 1);

        // printf("\norbit: "); for (int k=0; k<lenMatrix[i]; k++) printf("%i ", orbit[k]); printf("\n");
        // printf("\norbitRef: "); for (int k=0; k<lenMatrix[i]; k++) printf("%i ", orbitRef[k]); printf("\n");

        // apply orbit reference
        for (int j=0; j < lenMatrix[i]; j++) perm[orbitRef[j]] = orbit[j];
        // for (int j=0; j < lenMatrix[i]; j++) printf("%i <- %i", perm[orbitRef[j]], orbit[j]);

    }
    // printf("\nperm obtained: "); for (int k=0; k<numRows; k++) printf("%i ", perm[k]); printf("\n");

    free(orbit);
    free(orbitRef);

    // printf("\nPerm insides: "); for (int i=0; i<orbits.numRows; i++) printf("%i ", perm[i]); printf("\n");
}


void perm2orbit(int* perm, OrbitsData orbits){

    // get orbits data
    int* matrix = orbits.orbitsMatrix;
    int* lenMatrix = orbits.orbitsLen;
    int numRows = orbits.numRows;
    int numColumns = orbits.numColumns;

    int *orbit = malloc(sizeof(int) * numRows);
    int *check = malloc(sizeof(int) * numRows);

    for (int i=0; i < numRows; i++) check[i] = 0;
    for (int i=0; i < numRows; i++) lenMatrix[i] = 0;

    int j;
    int ii = 0;

    // Loop for all orbits list
    for (int i=0; i < numRows; i++){

        if (check[i] == 1) continue;

        j = 0;
        orbit[j] = perm[i];

        while (1) {
            lenMatrix[ii]++;

            matrix[two2one(ii, j, numColumns)] = orbit[j];
            check[orbit[j]] = 1;

            if  (perm[orbit[j]] == orbit[0]) break;

            orbit[j+1] = perm[orbit[j]];

            j++;
        }
        ii++;
    }

    // Free memory
    free(orbit);
    free(check);

}

int checkInList(int* list1, int* list2, int nList1, int nList2) {

    // printf("n_dimensions: %i %i\n", nList1, nList2);
    // printf("List 1: "); for (int i=0; i < nList1; i++) printf("%i ", list1[i]); printf("\n");
    // printf("List 2: "); for (int i=0; i < nList2; i++) printf("%i ", list2[i]); printf("\n");

    for (int i=0; i < nList1; i++){
        for (int j=0; j < nList2; j++){
            if (list1[i] == list2[j]) return 1;
        }
    }
    return 0;
}

void joinPerm(int* joinList, OrbitsData orbits, int numJoin) {

    // get data from structure
    int* orbitsMatrix = orbits.orbitsMatrix;
    int* orbitsLen = orbits.orbitsLen;
    int numRows = orbits.numRows;
    int numColumns = orbits.numColumns;

    // initialize
    int skip = 0;
    int placedFirst = 0;
    for (int i=0; i < numRows; i++){

        if (checkInList(orbitsMatrix + numColumns*(i-skip), joinList, orbitsLen[i-skip], numJoin)){
            if (placedFirst) {
                // printf("move up i: %i: value: %i\n", i, orbitsMatrix[two2one(i-skip, 0, numColumns)]);
                // move up matrix

                // printf("range: %i,  %i\n", i-skip, numRows-skip-1);

                for (int ii=i-skip; ii < numRows-skip-1; ii++)  {
                    // printf("merge %i into %i\n", ii+1, ii);
                    for (int j=0; j < orbitsLen[ii+1]; j++) {
                        orbitsMatrix[two2one(ii, j, numColumns)] = orbitsMatrix[two2one(ii+1, j, numColumns)];
                    }
                    orbitsLen[ii] = orbitsLen[ii+1];
                }
                orbitsLen[numRows - skip-1] = 0;
                skip++;

            } else {
                // put joinList in matrix
                // printf("placed joinList %i : %i\n", i, numJoin);
                for (int j=0; j < numJoin; j++) orbitsMatrix[two2one(i, j, numColumns)] = joinList[j];
                orbitsLen[i] = numJoin;
                placedFirst = 1;
            }
        }
    }
}



void breakPerm(OrbitsData orbits, int iBreak) {

    // get data from structure
    int* orbitsMatrix = orbits.orbitsMatrix;
    int* orbitsLen = orbits.orbitsLen;
    int numRows = orbits.numRows;
    int numColumns = orbits.numColumns;

    // printMatrix(orbits);
    int lenBreak = orbitsLen[iBreak];
    // printMatrix(orbits);
    // printf("len break %i: %i\n", iBreak, lenBreak);
    if (lenBreak == 1) {
        printf("break error!\n");
        exit(1);
        return ;
    }

    // roll to make space for broken orbit
    for (int i=numRows-lenBreak; i >= iBreak; i--){
        // printf("place %i -> %i\n", i, i+lenBreak-1);
        for (int j=0; j < orbitsLen[i]; j++) {
            orbitsMatrix[two2one(i+lenBreak-1, j, numColumns)] = orbitsMatrix[two2one(i, j, numColumns)];
        }
        // printf("perm index: %i, %i\n", i+lenBreak-1, orbitsLen[i]);
        orbitsLen[i+lenBreak-1] = orbitsLen[i];
    }

    // place broken orbit in rows
    for (int i=0; i < lenBreak; i++){
        // printf("move %i -> %i\n", i+iBreak , orbitsMatrix[two2one(iBreak, i, numColumns)]);
        orbitsMatrix[two2one(i+iBreak, 0, numColumns)] = orbitsMatrix[two2one(iBreak, i, numColumns)];
        orbitsLen[i+iBreak] = 1;
    }

}


void rollPerm(OrbitsData orbits, int iRoll, int nRoll) {

    // get data from structure
    int* orbitsMatrix = orbits.orbitsMatrix;
    int* orbitsLen = orbits.orbitsLen;
    int numColumns = orbits.numColumns;

    // roll orbit
    rollOrbit(orbitsMatrix + numColumns * iRoll, orbitsLen[iRoll], nRoll);

}


void mixPerm(OrbitsData orbits, int iOrbit1, int iOrbit2, int iPos1, int iPos2) {

    // get data from structure
    int* orbitsMatrix = orbits.orbitsMatrix;
    // int* orbitsLen = orbits.orbitsLen;
    int numColumns = orbits.numColumns;

    // swap values
    int backupFirst = orbitsMatrix[two2one(iOrbit1, iPos1, numColumns)];
    orbitsMatrix[two2one(iOrbit1, iPos1, numColumns)] = orbitsMatrix[two2one(iOrbit2, iPos2, numColumns)];
    orbitsMatrix[two2one(iOrbit2, iPos2, numColumns)] = backupFirst;

}

double evaluationPerm(double* distanceTable, int* permutation, int size) {
        double sum = 0;
        int j;
        for (int i=0; i<size; i++){
            j = permutation[i];
            sum += distanceTable[i * size + j];
        }

        return sum;

}


void updatePerm(OrbitsData orbits, int exponent) {

    // get data from structure
    int* orbitsMatrix = orbits.orbitsMatrix;
    int* orbitsLen = orbits.orbitsLen;
    int numColumns = orbits.numColumns;
    int numRows = orbits.numRows;

    int* singleOrbits = malloc(sizeof(int) * numRows);
    int* multiOrbits = malloc(sizeof(int) * numRows);

    int singleOrbitsNum = 0;
    int multiOrbitsNum = 0;

    for (int i=0; i<numRows; i++) {
        if (orbitsLen[i] < 1)  break;
        // printf("len: %i, %i\n", i, orbitsLen[i]);

        if (orbitsLen[i] > 1)  {
            // printf("multi: %i <- %i\n", multiOrbitsNum, i);
            multiOrbits[multiOrbitsNum] = i;
            multiOrbitsNum += 1;
        } else {
            // printf("single: %i <- %i\n", singleOrbitsNum, i);
            singleOrbits[singleOrbitsNum] = orbitsMatrix[two2one(i, 0, numColumns)];
            singleOrbitsNum += 1;
        }
    }

    /*
    printMatrix(orbits);

    printf("\nsingleOrbitsNum: %i\n", singleOrbitsNum);
    printf("multiOrbitsNum: %i\n", multiOrbitsNum);
    printf("singleOrbits list:"); for (int j=0; j<singleOrbitsNum; j++) printf(" %i",  singleOrbits[j]); printf("\n");
    printf("multiOrbits list:"); for (int j=0; j<multiOrbitsNum; j++) printf(" %i",  multiOrbits[j]); printf("\n");
    */

    // printMatrix(orbits);
    int sumTot = multiOrbitsNum + multiOrbitsNum + singleOrbitsNum;
    double pMix = (1.0 * multiOrbitsNum)/sumTot;
    double pJoin = (1.0 * singleOrbitsNum)/sumTot;
    double pBreak = (1.0 * multiOrbitsNum)/sumTot;
    // printf("Probabilities: %f %f %f\n", pMix, pJoin, pBreak);
    // printf("pjoin %i | %i\n", singleOrbitsNum, singleOrbitsNum);

    double r = (1.0*rand())/RAND_MAX;


    if (r <= pMix) {
        // Mix two orbits
        // printf("mix!\n");
        r = rand() % multiOrbitsNum;

        int iOrbit1 = rand() % multiOrbitsNum;
        int iOrbit2 = rand() % multiOrbitsNum;
        int iPos1 = rand() % orbitsLen[iOrbit1];
        int iPos2 = rand() % orbitsLen[iOrbit2];

        // printf("Mix chosen\n Select: %i %i %i %i\n", iOrbit1, iOrbit2, iPos1, iPos2);
        mixPerm(orbits, multiOrbits[iOrbit1], multiOrbits[iOrbit2], iPos1, iPos2);
        // printMatrix(orbits);
        // exit(0);
    }
    if (r > pMix && r <= (pJoin + pMix)) {
        // join orbits
        // printf("\nJoin! %i, %i\n", singleOrbitsNum, numColumns);
        if (singleOrbitsNum < exponent) return;

        int iChose;
        int numJoin = exponent;

        if (singleOrbitsNum >= numColumns) {
            iChose = rand() % 2;
            if (iChose) numJoin = numColumns;
        }

        int* joinList = malloc(sizeof(int) * numJoin);

        for (int i=0; i<numJoin; i++) {
            iChose = rand() % (singleOrbitsNum - i);
            joinList[i] = singleOrbits[iChose];
            singleOrbits[iChose] = singleOrbits[singleOrbitsNum-i-1];

        }

        joinPerm(joinList, orbits, numJoin);

        free(joinList);
    }

    if (r > (pJoin + pMix)) {

        // printMatrix(orbits);

        int iChose = rand() % multiOrbitsNum;
        int iBreak = multiOrbits[iChose];

        breakPerm(orbits, iBreak);

        // printMatrix(orbits);
        // printf("final\n");
        //exit(0);

    }

    // free memory
    free(singleOrbits);
    free(multiOrbits);

}

static PyObject* getPermutationSimple(PyObject* self, PyObject *arg, PyObject *keywords){

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

static PyObject* getPermutationAnnealing(PyObject* self, PyObject *arg, PyObject *keywords) {

    //  Interface with Python
    PyObject *distanceTable_obj;
    int order;
    int exponent=1;

    static char *kwlist[] = {"distance_table", "order", "exp", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "Oi|i", kwlist, &distanceTable_obj, &order, &exponent))  return NULL;

    PyObject *distanceTableArray = PyArray_FROM_OTF(distanceTable_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (distanceTableArray == NULL ) {
        Py_XDECREF(distanceTableArray);
        return NULL;
    }

    double *distanceTable = (double*)PyArray_DATA(distanceTableArray);

    int  nCoor1 = (int)PyArray_DIM(distanceTableArray, 0);
    int  nCoor2 = (int)PyArray_DIM(distanceTableArray, 1);
    int  n = (int)PyArray_DIM(distanceTableArray, 0);

    if (n != nCoor1 || n != nCoor2){
        Py_DECREF(distanceTableArray);
        PyErr_SetString(PyExc_TypeError, "Dimensions error ");
        return (PyObject *) NULL;
    }

    PyArrayObject *perm_obj;
    npy_intp dims[]={n};
    perm_obj=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);
    int *perm = (int*)PyArray_DATA(perm_obj);

    srand(time(NULL));   // Initialization, should only be called once.

    // Initial params
    // printf("\n\norder/exponent: %i %i\n", order, exponent);

    // initialize permutation
    for (int i=0; i<n; i++ ) perm[i] = i;

    // initialize orbits structure
    OrbitsData orbits;
    orbits.orbitsMatrix = malloc(sizeof(int)*order*n);
    orbits.orbitsLen = malloc(sizeof(int)*n);
    orbits.numRows = n;
    orbits.numColumns = order;

    int* newPerm = malloc(sizeof(int)*n);
    int* oldPerm = malloc(sizeof(int)*n);

    // initialize
    for (int j=0; j<orbits.numRows; j++) oldPerm[j] = perm[j];

    double newEvaluation;
    double prob;

    double temp = 100;
    perm2orbit(oldPerm, orbits);
    double evaluation = evaluationPerm(distanceTable, oldPerm, n);
    double bestEvaluation = evaluation;

    // printf("-> "); for(int i=0; i<n; i++) printf(" %i", perm[i]); printf("\n");

    int nSteps = 100;
    for (int i=0; i<nSteps; i++){
        // printf("step: %i\n", i);
        //printf("-> "); for(int i=0; i<n; i++) printf(" %i", perm[i]); printf("\n");

        updatePerm(orbits, exponent);
        orbit2perm(newPerm, orbits);

        newEvaluation = evaluationPerm(distanceTable, newPerm, n);

        // keep the best of the best
        if (newEvaluation < bestEvaluation) {
            for (int j=0; j<orbits.numRows; j++) perm[j] = newPerm[j];
            bestEvaluation = newEvaluation;
        }

        if (newEvaluation >= evaluation) {
            prob = 1;
        } else {
            prob = exp((newEvaluation- evaluation)/temp);
        }

        if  ((1.0*rand())/RAND_MAX >= prob) {
            for (int j=0; j<orbits.numRows; j++) oldPerm[j] = newPerm[j];
            evaluation = newEvaluation;
        }

        temp *= 0.9;


        perm2orbit(oldPerm, orbits);

        // printMatrix(orbits);
        // printf("Evaluation %f\n", evaluation);

    }

    free(newPerm);
    free(oldPerm);

    // printf("_perm final: "); for (int i=0; i<orbits.numRows; i++) printf("%i ", perm[i]); printf("\n");
    // printf("_Evaluation final %f \n", evaluation);

    // Free c memory
    free(orbits.orbitsMatrix);
    free(orbits.orbitsLen);

    // Free python memory
    Py_DECREF(distanceTableArray);

    return(PyArray_Return(perm_obj));

}


// Function to swap two elements in an array
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function to generate all permutations of an array
void generatePermutations(int *perm, int start, int end, int exp, double*distanceTable, OrbitsData orbits, double*maxEval, int*maxPerm) {

    if (start == end) {
        // Print the current permutation
        // for (int i = 0; i <= end; i++) printf("%d ", perm[i]); printf("\n");
        double evaluation;

        perm2orbit(perm, orbits);
        int numHighOrbits = 0;
        for (int i=0; i<orbits.numColumns; i++){
            if (orbits.orbitsLen[i] == 0) break;
            if (orbits.orbitsLen[i] != orbits.numColumns && orbits.orbitsLen[i] != 1 && orbits.orbitsLen[i] != exp) {
                numHighOrbits = -1;
                break;
            }
            numHighOrbits++;
        }
        // printf("perm outside: "); for (int i=0; i<orbits.numRows; i++) printf("%i ", perm[i]); printf("\n");
        // printf("numHighOrbits: %i\n", numHighOrbits);
        if (numHighOrbits > 0) {
            // printf("perm inside: "); for (int i=0; i<orbits.numRows; i++) printf("%i ", perm[i]); printf("\n");

            evaluation = evaluationPerm(distanceTable, perm, end+1);
            // printf("eval_test: %f / %f\n", evaluation, end);
            // printMatrix(orbits);
            if (*maxEval > evaluation) {
                *maxEval = evaluation;

                for (int k=0; k<orbits.numRows; k++) maxPerm[k] = perm[k];
            }
        }

    } else {
        // Recursive permutation generation
        for (int i = start; i <= end; i++) {
            // Swap the current element with itself and all subsequent elements
            swap(&perm[start], &perm[i]);

            // Recur for the remaining elements
            generatePermutations(perm, start + 1, end, exp, distanceTable, orbits, maxEval, maxPerm);

            // Backtrack: Undo the swap to restore the original array
            swap(&perm[start], &perm[i]);
        }
    }
}


static PyObject* getPermutationBruteForce(PyObject* self, PyObject *arg, PyObject *keywords) {

    //  Interface with Python
    PyObject *distanceTable_obj;
    int order;
    int exponent=1;

    static char *kwlist[] = {"distance_table", "order", "exp", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "Oi|i", kwlist, &distanceTable_obj, &order, &exponent))  return NULL;

    PyObject *distanceTableArray = PyArray_FROM_OTF(distanceTable_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (distanceTableArray == NULL ) {
        Py_XDECREF(distanceTableArray);
        return NULL;
    }

    double *distanceTable = (double*)PyArray_DATA(distanceTableArray);

    int  nCoor1 = (int)PyArray_DIM(distanceTableArray, 0);
    int  nCoor2 = (int)PyArray_DIM(distanceTableArray, 1);
    int  n = (int)PyArray_DIM(distanceTableArray, 0);

    if (n != nCoor1 || n != nCoor2){
        Py_DECREF(distanceTableArray);
        PyErr_SetString(PyExc_TypeError, "Dimensions error ");
        return (PyObject *) NULL;
    }

    PyArrayObject *perm_obj;
    npy_intp dims[]={n};
    perm_obj=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);
    int *perm = (int*)PyArray_DATA(perm_obj);

    srand(time(NULL));   // Initialization, should only be called once.

    // Initial params
    //printf("order/exponent: %i %i\n", order, exponent);

    // initialize permutation
    for (int i=0; i<n; i++)perm[i] = i;

    // initialize orbits
    OrbitsData orbits;
    orbits.orbitsMatrix = malloc(sizeof(int)*order*n);
    orbits.orbitsLen = malloc(sizeof(int)*n);
    orbits.numRows = n;
    orbits.numColumns = order;

    // initialize
    for (int j=0; j<orbits.numRows; j++) perm[j] = j;

    double evaluation;

    perm2orbit(perm, orbits);
    // printMatrix(orbits);


    // Input the elements of the vector
    int* newPerm = malloc(sizeof(int)*n);
    for (int j=0; j<orbits.numRows; j++) newPerm[j] = j;
    double maxEval = evaluationPerm(distanceTable, newPerm, n);

    // Generate and print all permutations
    // printf("\nAll permutations of the vector:\n");

    generatePermutations(newPerm, 0, n - 1, exponent, distanceTable, orbits, &maxEval, perm);

    free(newPerm);

    // printf("maxEval out: %f\n", maxEval);


    // printf("\n\nperm final: "); for (int i=0; i<orbits.numRows; i++) printf("%i ", perm[i]); printf("\n");
    evaluation = evaluationPerm(distanceTable, perm, n);
    // printf("Evaluation final %f \n", evaluation);

    perm2orbit(perm, orbits);
    // printMatrix(orbits);
    // Free c memory
    free(orbits.orbitsMatrix);
    free(orbits.orbitsLen);

    // Free python memory
    Py_DECREF(distanceTableArray);

    return(PyArray_Return(perm_obj));

}


static PyObject* fixPermutationSimple(PyObject* self, PyObject *arg, PyObject *keywords) {

    //  Interface with Python
    PyObject *distanceTable_obj;
    PyObject *intialPermutation_obj;
    int order;
    int exponent=1;

    static char *kwlist[] = {"distance_table", "permutation", "order", "exp", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOi|i", kwlist, &distanceTable_obj, &intialPermutation_obj,
    &order, &exponent))  return NULL;

    PyObject *distanceTableArray = PyArray_FROM_OTF(distanceTable_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *permutationArray = PyArray_FROM_OTF(intialPermutation_obj, NPY_INT, NPY_IN_ARRAY);

    if (distanceTableArray == NULL || permutationArray == NULL) {
        Py_XDECREF(distanceTableArray);
        Py_XDECREF(permutationArray);
        return NULL;
    }

    double *distanceTable = (double*)PyArray_DATA(distanceTableArray);
    int *perm = (double*)PyArray_DATA(permutationArray);

    int  nCoor1 = (int)PyArray_DIM(distanceTableArray, 0);
    int  nCoor2 = (int)PyArray_DIM(distanceTableArray, 1);
    int  n = (int)PyArray_DIM(permutationArray, 0);

    if (n != nCoor1 || n != nCoor2){
        Py_DECREF(distanceTableArray);
        PyErr_SetString(PyExc_TypeError, "Dimensions error ");
        return (PyObject *) NULL;
    }

    // Initial params
    printf("order/exponent: %i %i\n", order, exponent);

    // initialize permutation
    printf("-> "); for(int i=0; i<n; i++) printf(" %i", perm[i]); printf("\n");

    // initialize orbits
    OrbitsData orbits;
    orbits.orbitsMatrix = malloc(sizeof(int)*order*n);
    orbits.orbitsLen = malloc(sizeof(int)*n);
    orbits.numRows = n;
    orbits.numColumns = order;

    for (int i=0; i<n; i++) {
        if (orbits.orbitsLen[i] <= 0) break;
        if (orbits.orbitsLen[i] > 1) {
            for (int j=0; j<orbits.orbitsLen[i]-order; j++) {
                for (int k=0; k<orbits.orbitsLen[i]; k++) {
                    // partialBreakPerm(orbits, i, k);


                }

            }
        }
    }



    perm2orbit(perm, orbits);
    printMatrix(orbits);

    // Free c memory
    free(orbits.orbitsMatrix);
    free(orbits.orbitsLen);

    // Free python memory
    Py_DECREF(distanceTableArray);

    return(PyArray_Return(intialPermutation_obj));

    /*
    // Create list
    PyObject* python_val = PyList_New(0);

    int value_len;
    for (int i=0; i<n; i++) {
        value_len = orbits.orbitsLen[i];
        if (value_len > 0) PyList_Append(python_val, Py_BuildValue("i", value_len));
    }


    PyObject* python_val = PyList_New(n);
    for (int i = 0; i < n; i++)
    {
        int r = rand() % 100;
        PyObject* python_int = Py_BuildValue("i", r);
        PyObject* python_int_2 = Py_BuildValue("i", r+1);

        PyObject* python_inside = PyList_New(2);
        PyList_SetItem(python_inside, 0, python_int);
        PyList_SetItem(python_inside, 1, python_int_2);

        PyList_SetItem(python_val, i, python_inside);
    }
    */

    //return python_val;
    // return(PyArray_Return(permutation_obj));


}

static PyObject* validatePermutation(PyObject* self, PyObject *arg, PyObject *keywords) {

    //  Interface with Python
    PyObject *permutation_obj;
    int order=1;
    int determinant=1;

    static char *kwlist[] = {"permutation", "order", "determinant", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O|ii", kwlist, &permutation_obj, &order, &determinant))  return NULL;

    PyObject *permutationArray = PyArray_FROM_OTF(permutation_obj, NPY_INT, NPY_IN_ARRAY);

    if (permutationArray == NULL) {
        Py_XDECREF(permutationArray);
        return NULL;
    }

    int *perm = (int*)PyArray_DATA(permutationArray);
    int  n = (int)PyArray_DIM(permutationArray, 0);

    // initialize orbits
    OrbitsData orbits;
    orbits.orbitsMatrix = malloc(sizeof(int)*n*n);
    orbits.orbitsLen = malloc(sizeof(int)*n);
    orbits.numRows = n;
    orbits.numColumns = n;

    perm2orbit(perm, orbits);
    // printMatrix(orbits);

    int value_len;
    for (int i=0; i<n; i++) {
        value_len = orbits.orbitsLen[i];
        if (value_len > 0){
            if (value_len == 1) continue;
            if (value_len == order) continue;
            if (determinant < 0){
                if (value_len == 2) continue;
                if (value_len == 2 * order) continue;
            }

            // Free memory
            free(orbits.orbitsMatrix);
            free(orbits.orbitsLen);
            Py_RETURN_FALSE;
        }
    };

    // Free memory
    free(orbits.orbitsMatrix);
    free(orbits.orbitsLen);

    Py_RETURN_TRUE;
}

