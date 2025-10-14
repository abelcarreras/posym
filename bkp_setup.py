from setuptools import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]

def check_compiler():
    import subprocess
    output = subprocess.Popen(['gcc'], stderr=subprocess.PIPE).communicate()[1]
    if b'clang' in output:
        return 'clang'
    if b'gcc' in output:
        return 'gcc'


if check_compiler() == 'clang':
    integrals = Extension('posym.integrals',
                          extra_compile_args=['-std=c99'],
                          include_dirs=include_dirs_numpy,
                          sources=['c/integrals.c'])

else:
    print ('openmp is used')
    integrals = Extension('posym.integrals',
                          extra_compile_args=['-std=c99', '-fopenmp'],
                          extra_link_args=['-lgomp'],
                          include_dirs=include_dirs_numpy,
                          sources=['c/integrals.c'])

permutations = Extension('posym.permutation.permutations',
                         extra_compile_args=['-std=c99'],
                         include_dirs=include_dirs_numpy,
                         sources=['c/permutations.c'])


setup(ext_modules=[integrals, permutations])
